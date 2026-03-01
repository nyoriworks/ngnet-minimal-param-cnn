"""
Train NGNet-6 on CIFAR-10, CIFAR-100, or Tiny ImageNet.

Usage:
    python train.py --dataset cifar100 --epochs 50
    python train.py --dataset cifar10 --epochs 50
    python train.py --dataset tiny-imagenet --epochs 50 --data-dir ./tiny-imagenet-200
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse
import time
import os

from ngnet import NGNet6
from ngconv import NGConv


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    correct = total = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        use_amp = scaler is not None
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(data)
            loss = criterion(out, target)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        correct += out.argmax(1).eq(target).sum().item()
        total += target.size(0)
    return 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, device, scaler=None):
    model.eval()
    correct = total = 0
    use_amp = scaler is not None
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            correct += model(data).argmax(1).eq(target).sum().item()
        total += target.size(0)
    return 100.0 * correct / total


def setup_tiny_imagenet(data_dir):
    """Reorganize Tiny ImageNet val directory if needed."""
    val_dir = os.path.join(data_dir, "val")
    val_img_dir = os.path.join(val_dir, "images")
    if os.path.isdir(val_img_dir):
        print("Reorganizing Tiny ImageNet val set...")
        with open(os.path.join(val_dir, "val_annotations.txt")) as f:
            for line in f:
                parts = line.strip().split("\t")
                fname, cls = parts[0], parts[1]
                cls_dir = os.path.join(val_dir, cls)
                os.makedirs(cls_dir, exist_ok=True)
                src = os.path.join(val_img_dir, fname)
                dst = os.path.join(cls_dir, fname)
                if os.path.exists(src) and not os.path.exists(dst):
                    os.rename(src, dst)
        try:
            os.rmdir(val_img_dir)
        except OSError:
            pass


def main():
    parser = argparse.ArgumentParser(description="Train NGNet-6")
    parser.add_argument("--dataset", type=str, default="cifar100",
                        choices=["cifar10", "cifar100", "tiny-imagenet"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--data-dir", type=str, default="./data")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # Dataset setup
    if args.dataset == "cifar10":
        stats = ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        num_classes, img_size = 10, 32
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), transforms.Normalize(*stats),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.15))])
        test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)])
        train_ds = datasets.CIFAR10(args.data_dir, True, download=True, transform=train_tf)
        test_ds = datasets.CIFAR10(args.data_dir, False, transform=test_tf)

    elif args.dataset == "cifar100":
        stats = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        num_classes, img_size = 100, 32
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), transforms.Normalize(*stats),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.15))])
        test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)])
        train_ds = datasets.CIFAR100(args.data_dir, True, download=True, transform=train_tf)
        test_ds = datasets.CIFAR100(args.data_dir, False, transform=test_tf)

    elif args.dataset == "tiny-imagenet":
        stats = ((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
        num_classes, img_size = 200, 64
        setup_tiny_imagenet(args.data_dir)
        train_tf = transforms.Compose([
            transforms.RandomCrop(64, padding=8), transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), transforms.Normalize(*stats),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.15))])
        test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)])
        train_ds = datasets.ImageFolder(os.path.join(args.data_dir, "train"), transform=train_tf)
        test_ds = datasets.ImageFolder(os.path.join(args.data_dir, "val"), transform=test_tf)

    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    test_loader = DataLoader(test_ds, 128, shuffle=False, num_workers=2)

    # Model
    model = NGNet6(num_classes=num_classes, img_size=img_size).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"NGNet-6 ({args.dataset}): {params:,} params")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best = 0.0
    t_start = time.time()

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tr = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        te = evaluate(model, test_loader, device, scaler)
        scheduler.step()
        best = max(best, te)
        elapsed = time.time() - t0

        if ep % 10 == 0 or ep == 1 or ep == args.epochs:
            total = time.time() - t_start
            print(f"Ep {ep:3d}/{args.epochs}: Train={tr:.1f}% Test={te:.1f}% "
                  f"(best={best:.1f}%) {elapsed:.0f}s [{total/60:.0f}m]")

    total_time = time.time() - t_start
    print(f"\nDone. Best: {best:.2f}% | Params: {params:,} | Time: {total_time/60:.1f}min")


if __name__ == "__main__":
    main()
