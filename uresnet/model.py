# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F

from .parts import *

class LighterUResNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(LighterUResNet, self).__init__()
        self.inc = inconv(input_channels, 12)
        self.down1 = down(12, 24)
        self.down2 = down(24, 48)
        self.down3 = down(48, 48)
        self.up2 = up(96, 24)
        self.up3 = up(48, 12)
        self.up4 = up(24, 12)
        self.outc = outconv(12, output_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        # return F.sigmoid(x)
        return torch.sigmoid(x)
