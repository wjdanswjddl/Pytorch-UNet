import torch
import torch.nn as nn
import torch.nn.functional as f


class Downsample2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose2d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv2dBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, kernel_size=5, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish()
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, kernel_size=5):
        super().__init__()
        self.blocks = nn.Sequential(
            Conv2dBlock(inp_channels, out_channels, kernel_size),
            Conv2dBlock(out_channels, out_channels, kernel_size)
        )
        self.residual_conv = nn.Conv2d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x):
        out = self.blocks(x) + self.residual_conv(x)
        return out
