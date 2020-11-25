#!/usr/bin/env python

import argparse
import os

import numpy as np
import torch

from unet import UNet
from uresnet import UResNet
from nestedunet import NestedUNet

from eval_util import eval_dice, eval_loss, eval_eff_pur
from utils import h5_utils as h5u
from matplotlib import cm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which is stored the model"
                             " (default : 'MODEL.pth')")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=False)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+',
                        help='filenames of ouput images')
    parser.add_argument('--gpu', '-g', action='store_true',
                        help="Use cuda",
                        default=True)
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--range', '-r', type=int, nargs='+',
                        help="Event range to be processed",
                        default=0)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=0.5)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    torch.set_num_threads(1)

    # im_tags = ['frame_tight_lf0', 'frame_loose_lf0'] #lt
    im_tags = ['frame_loose_lf0', 'frame_mp2_roi0', 'frame_mp3_roi0']    # l23
    # im_tags = ['frame_loose_lf0', 'frame_tight_lf0', 'frame_mp2_roi0', 'frame_mp3_roi0']    # lt23

    ma_tags = ['frame_ductor0']

    print("Loading model {}".format(args.model))
    if args.model.endswith(".ts"):
        net = torch.jit.load(args.model)
        if args.gpu:
            net.cuda()
        else:
            net.cpu()
    else:
        net = UNet(len(im_tags), 1)
        # net = UResNet(len(im_tags), 1)
        # net = NestedUNet(len(im_tags), 1)

        if args.gpu:
            net.cuda()
            net.load_state_dict(torch.load(args.model))
        else:
            net.cpu()
            net.load_state_dict(torch.load(args.model, map_location='cpu'))

    print("Model loaded !")

    dir_out = 'out-eval/'
    # v plane
    # eval_labels = [
    #     '75-75',
    #     '80-80',
    #     '82-82',
    #     '85-85',
    #     '87-75',
    #     '87-85',
    #     '87-87',
    #     ]
    
    # u plane
    eval_labels = [
        '75-75',
        '80-80',
        '82-82',
        '85-85',
        # '75-87',
        # '85-87',
        '87-87',
        ]
    eval_imgs = []
    eval_masks = []
    for label in eval_labels:
        eval_imgs.append('eval-jinst-init-sub/eval-'+label+'/g4-rec-0.h5')
        eval_masks.append('eval-jinst-init-sub/eval-'+label+'/g4-tru-0.h5') 
            
    rebin = [1, 10]
    # x_range = [0, 800]
    x_range = [800, 1600]
    y_range = [0, 600]
    z_scale = 4000
    truth_th = 100

    eval_data = []
    for i in range(len(eval_imgs)):
        id_eval = [100]
        eval_data.append(
            zip(
                h5u.get_chw_imgs(eval_imgs[i], id_eval,   im_tags, rebin, x_range, y_range, z_scale),
                h5u.get_masks(eval_masks[i],   id_eval,   ma_tags, rebin, x_range, y_range, truth_th)
            )
        )

    outfile_ep = open(dir_out+'/'+args.output[0]+'.csv','w')
    for label, data in zip(eval_labels, eval_data):
        print(label)
        ep = eval_eff_pur(net, data, 0.5, args.gpu)
        print('{}, {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(label, ep[0], ep[1], ep[2], ep[3]))
        print('{}, {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(label, ep[0], ep[1], ep[2], ep[3]), file=outfile_ep)