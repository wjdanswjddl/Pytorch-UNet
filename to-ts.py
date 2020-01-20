#!/usr/bin/env python

import argparse
import os

import torch
import torchvision

from unet import UNet


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which is stored the model"
                              " (default : 'MODEL.pth')")

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    
    net = UNet(n_channels=3, n_classes=1)

    example = torch.rand(1, 3, 800, 600)

    gpu = True
    
    if gpu:
        net.cuda()
        net.load_state_dict(torch.load(args.model))
        sm = torch.jit.trace(net, example.cuda())
        output = net(example.cuda())
        print(output[0][0][0])
    else:
        net.cpu()
        net.load_state_dict(torch.load(args.model, map_location='cpu'))
        sm = torch.jit.trace(net, example)
        output = net(example)
        print(output[0][0][0])


    sm.save('ts-model.pt')