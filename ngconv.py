"""
NGConv: Neural Gated Convolution

A parameter-efficient convolutional block that splits input channels into
4 specialized phases, applying expensive spatial convolution to only 25%
of channels while maintaining full learning capacity through lightweight
channel mixing and attention.

Key properties:
    - ~18% FLOPs of Standard Conv (same channel size)
    - ~same parameter count as PConv (FasterNet)
    - Residual connection when in_ch == out_ch
    - Drop-in replacement for standard Conv+BN+Act blocks

Reference:
    https://www.kaggle.com/code/nyoriworks/ngnet-minimal-param-cnn
    https://github.com/nyoriworks/ngnet-minimal-param-cnn
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedActivation(nn.Module):
    """Learnable interpolation between bounded tanh and unbounded identity.

    f(x) = (1 - σ(γ)) · tanh(α·x)/α + σ(γ) · x

    Early training uses the bounded tanh branch for stability;
    as training progresses, the gate opens toward identity for expressiveness.
    """
    SCALE = (4.0 / 5.0) * math.pi

    def __init__(self, channels: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gamma)
        return (1 - g) * torch.tanh(self.SCALE * x) / self.SCALE + g * x


class NGConv(nn.Module):
    """Neural Gated Convolution.

    Splits input into 4 phases (each ~25% of channels):
        phase_x: Conv2d k×k  — spatial feature extraction (the only heavy op)
        phase_y: 1×1 Conv + GatedActivation — channel mixing (light)
        phase_z: GAP → FC → FC — channel attention, SE-style (minimal)
        phase_τ: identity — zero FLOPs, zero params

    All phases are concatenated and mixed through a 1×1 cross_mixer.

    Args:
        in_ch:  Input channels
        out_ch: Output channels
        k:      Spatial kernel size for phase_x (default: 3)
        ratio:  Fraction of channels per phase (default: 0.25)
    """
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, ratio: float = 0.25):
        super().__init__()
        self.dim_x = max(1, int(in_ch * ratio))
        self.dim_y = max(1, int(in_ch * ratio))
        self.dim_z = max(1, int(in_ch * ratio))
        self.dim_tau = in_ch - (self.dim_x + self.dim_y + self.dim_z)
        self.splits = [self.dim_x, self.dim_y, self.dim_z, self.dim_tau]

        # Phase X: spatial convolution (only heavy operation)
        self.phase_x = nn.Conv2d(self.dim_x, self.dim_x, k, padding=k // 2, bias=False)

        # Phase Y: channel mixing with gated activation
        self.phase_y_mix = nn.Conv2d(self.dim_y, self.dim_y, 1, bias=False)
        self.phase_y_act = GatedActivation(self.dim_y)

        # Phase Z: squeeze-excitation style channel attention
        dz2 = max(1, self.dim_z // 2)
        self.phase_z_fc = nn.Sequential(
            nn.Linear(self.dim_z, dz2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dz2, self.dim_z, bias=False),
            nn.Tanh(),
        )

        # Cross-phase mixer: 1×1 conv lets all phases interact
        self.cross_mixer = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

        self.gate = nn.Parameter(torch.tensor([0.1]))
        self.has_residual = in_ch == out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        px, py, pz, ptau = torch.split(x, self.splits, dim=1)

        # Spatial features (25% of channels, k×k conv)
        px = self.phase_x(px)

        # Channel mixing (25%, 1×1 conv + gated activation)
        py = self.phase_y_act(self.phase_y_mix(py)) * torch.sigmoid(self.gate)

        # Channel attention (25%, SE-style global context)
        b, c = pz.shape[:2]
        att = F.adaptive_avg_pool2d(pz, 1).view(b, c)
        pz = pz * (1.0 + self.phase_z_fc(att).view(b, c, 1, 1))

        # phase_τ: unchanged (25%, zero FLOPs)

        out = self.cross_mixer(torch.cat([px, py, pz, ptau], dim=1))
        if self.has_residual:
            out = out + x
        return out
