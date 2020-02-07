#!/usr/bin/env python

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.pool =nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 1, 3, stride=2, padding=1),
            nn.Conv2d(1, 1, 3, stride=1, padding=1)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(1, 1, 1, stride=2, padding=0),
        )
    
    def forward(self, x):
        # x = self.pool(x)
        # x = self.up(x)
        # x = self.conv(x)
        # x = F.pad(x, (-1,0, 1, 0))
        return x

if __name__ == "__main__":
    
    # x = torch.rand(1, 1, 8, 5)
    x = torch.tensor([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]], dtype=torch.float).unsqueeze(0).unsqueeze(0)
    # x = torch.tensor([[1, 2], [4, 5], [7, 8]], dtype=torch.float).unsqueeze(0).unsqueeze(0)

    net = NN()
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('params = ', params)

    print(x.size())
    x = net(x)
    print(x.size())




