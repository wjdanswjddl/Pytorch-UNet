import torch
import torch.nn as nn
import torch.nn.functional as F
from .part import *


class UNet(nn.Module):
    def __init__(self, inp_channels, dim=128, dim_mults=(1, 2, 4, 8)):
        super().__init__()
        dims = [inp_channels, *[dim * m for m in dim_mults]]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            print(dim_out)
            self.downs.append(nn.ModuleList([
                ResidualBlock(dim_in, dim_out),
                ResidualBlock(dim_out, dim_out),
                Downsample2d(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResidualBlock(mid_dim, mid_dim)
        self.mid_block2 = ResidualBlock(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(nn.ModuleList([
                ResidualBlock(dim_out * 2, dim_in),
                ResidualBlock(dim_in, dim_in),
                Upsample2d(dim_in) if not is_last else nn.Identity()
            ]))

        self.final_conv = nn.Sequential(
            Conv2dBlock(dim, dim, kernel_size=5),
            nn.Conv2d(dim, 1, 1)
        )

    def forward(self, x):
        B, C, W, H = x.size()
        w_pad = 8 * (W // 8 + 1) - W
        h_pad = 8 * (H // 8 + 1) - H
        w_padding = torch.zeros([B, C, w_pad, H], dtype=torch.float32, device=x.device)
        x = torch.cat([x, w_padding], dim=2)
        h_padding = torch.zeros([B, C, W + w_pad, h_pad], dtype=torch.float32, device=x.device)
        x = torch.cat([x, h_padding], dim=3)
        h = []
        for resnet, resnet2, downsample in self.downs:
            x = resnet(x)
            x = resnet2(x)
            h.append(x)
            x = downsample(x)
        
        x = self.mid_block1(x)
        x = self.mid_block2(x)

        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x)
            x = resnet2(x)
            x = upsample(x)

        x = self.final_conv(x)
        x = x[:, :, :W, :H]

        return torch.sigmoid(x)
