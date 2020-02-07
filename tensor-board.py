#!/usr/bin/env python

import argparse
import os

import torch
import torchvision

from unet import UNet

from torch.utils.tensorboard import SummaryWriter

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which is stored the model"
                              " (default : 'MODEL.pth')")

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter('tblog/test1')
        
    net = UNet(n_channels=3, n_classes=1)

    images = torch.rand(2, 3, 800, 600)

    img_grid = torchvision.utils.make_grid(images)


    writer.add_image('rand_imgs', img_grid)

    net.load_state_dict(torch.load(args.model))

    writer.add_graph(net, images)
    writer.close()