#!/usr/bin/env python

import argparse
import os
import numpy as np

import torch
import torchvision

from unet import UNet
from uresnet import UResNet
from nestedunet import NestedUNet


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which is stored the model"
                              " (default : 'MODEL.pth')")
    parser.add_argument('--gpu', '-g', action='store_true',
                        help="Use cuda version of the net",
                        default=False)

    return parser.parse_args()
def count_params(net):
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('params = ', params)

if __name__ == "__main__":
    args = get_args()
    
    # net = UNet(3, 1)
    net = NestedUNet(3, 1)

    # count_params(net)

    example = torch.rand(1, 3, 800, 600)
    
    if args.gpu:
        net.cuda()
        net.load_state_dict(torch.load(args.model))
        sm = torch.jit.trace(net, example.cuda())
        output = net(example.cuda())
        # print(output[0][0][0])
    else:
        net.cpu()
        net.load_state_dict(torch.load(args.model, map_location='cpu'))
        sm = torch.jit.trace(net, example)
        output = net(example)
        # print(output[0][0][0])


    sm.save('ts-model.ts')