"""
NGNet-6: 6-layer Crescendo Architecture with NGConv

Channel pattern: 64 → 96 → 48 → 120 → 64 → 120 → 200
Each compression cycle is stronger than the previous,
forcing increasingly abstract feature representations.

Results (same crescendo channels, 50 epochs):
    CIFAR-100:      NGNet-6 118K → 65.07% | PConvNet-6 112K → 64.16% | StdNet-6 677K → 66.77%
    Tiny ImageNet:  NGNet-6 143K → 48.90% | PConvNet-6 136K → 48.50% | StdNet-6 702K → 52.16%

Usage:
    from ngnet import NGNet6, NGConv
    model = NGNet6(num_classes=100, img_size=32)  # CIFAR-100
    model = NGNet6(num_classes=200, img_size=64)  # Tiny ImageNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ngconv import NGConv


# Crescendo channel pattern
CHANNELS = [64, 96, 48, 120, 64, 120, 200]

# Spatial downsampling points
DOWN_AT = {2: 16, 4: 10}


class NGNet6(nn.Module):
    """6-layer crescendo network using NGConv blocks.

    Args:
        num_classes: Number of output classes
        img_size:    Input image size (32 for CIFAR, 64 for Tiny ImageNet)
    """
    def __init__(self, num_classes: int = 100, img_size: int = 32):
        super().__init__()
        chs = CHANNELS

        # Stem: adapt kernel/stride for input resolution
        stem_k = 5 if img_size <= 32 else 7
        stem_s = 1 if img_size <= 32 else 2
        self.stem = nn.Sequential(
            nn.Conv2d(3, chs[0], stem_k, stride=stem_s, padding=stem_k // 2, bias=False),
            nn.BatchNorm2d(chs[0]),
            nn.GELU(),
        )

        # 6 NGConv layers with crescendo channel pattern
        self.blocks = nn.ModuleList()
        for i in range(len(chs) - 1):
            in_c, out_c = chs[i], chs[i + 1]
            k = 5 if in_c <= 48 else 3
            self.blocks.append(NGConv(in_c, out_c, k=k))

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(chs[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for i, block in enumerate(self.blocks):
            if i in DOWN_AT and x.shape[2] > DOWN_AT[i]:
                x = F.adaptive_avg_pool2d(x, (DOWN_AT[i], DOWN_AT[i]))
            x = block(x)
        return self.fc(self.pool(x).flatten(1))


if __name__ == "__main__":
    # Quick sanity check
    for nc, sz, ds in [(10, 32, "CIFAR-10"), (100, 32, "CIFAR-100"), (200, 64, "Tiny ImageNet")]:
        model = NGNet6(num_classes=nc, img_size=sz)
        params = sum(p.numel() for p in model.parameters())
        x = torch.randn(2, 3, sz, sz)
        y = model(x)
        print(f"NGNet-6 ({ds}): {params:,} params, input {tuple(x.shape)} → output {tuple(y.shape)}")
