#!/usr/bin/env python

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image

from unet import UNet
from uresnet import UResNet
from nestedunet import NestedUNet

from utils import resize_and_crop, normalize, split_img_into_squares, hwc_to_chw, merge_masks
# from utils import dense_crf
from utils import plot_img_and_mask
from utils import h5_utils as h5u
from matplotlib import cm

from torchvision import transforms

def predict_img(net,
                full_img,
                scale_factor=0.5,
                out_threshold=0.5,
                use_dense_crf=True,
                use_gpu=False):
    
    # eval mode fixes BN and dropout, which yields bad results
    # net.eval()
    
    img_tensor = torch.from_numpy(hwc_to_chw(full_img))
    if use_gpu:
        img_tensor = img_tensor.cuda()

    with torch.no_grad():
        input = img_tensor.unsqueeze(0)
        print ("input.shape: ", input.shape)
        full_mask = net(input).squeeze().cpu().numpy()
        print ("full_mask.shape: ", full_mask.shape)

    if out_threshold < 0:
      return full_mask
      
    return full_mask > out_threshold


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which is stored the model"
                             " (default : 'MODEL.pth')")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='filenames of ouput images')
    parser.add_argument('--cpu', '-c', action='store_true',
                        help="Do not use the cuda version of the net",
                        default=False)
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--no-crf', action='store_true',
                        help="Do not use dense CRF postprocessing",
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

def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            # out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
            out_files.append("{}_OUT{}".format(pathsplit[0], '.jpg'))
    elif len(in_files) != len(args.output):
        print("Error : Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files

def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))

if __name__ == "__main__":
    args = get_args()
    assert len(args.range)==2, "range needs 2 inputs"

    in_files = args.input
    out_files = get_output_filenames(args)

    torch.set_num_threads(1)

    # im_tags = ['frame_tight_lf0', 'frame_loose_lf0'] #lt
    im_tags = ['frame_loose_lf0', 'frame_mp2_roi0', 'frame_mp3_roi0']    # l23
    # im_tags = ['frame_loose_lf0', 'frame_tight_lf0', 'frame_mp2_roi0', 'frame_mp3_roi0']    # lt23

    print("Loading model {}".format(args.model))
    if args.model.endswith(".ts"):
        net = torch.jit.load(args.model)
        if not args.cpu:
            net.cuda()
        else:
            net.cpu()
    else:
        net = UNet(len(im_tags), 1)
        # net = UResNet(len(im_tags), 1)
        # net = NestedUNet(len(im_tags), 1)

        if not args.cpu:
            net.cuda()
            net.load_state_dict(torch.load(args.model))
        else:
            net.cpu()
            net.load_state_dict(torch.load(args.model, map_location='cpu'))

    print("Model loaded !")

    for i, fn in enumerate(in_files):
        print("\nPredicting image {} ...".format(fn))
       

        events = list(np.arange(args.range[0], args.range[1]))
        for event in events:
            # img = h5u.get_hwc_img(fn, event, im_tags, [1, 10], [0, 800], [0, 600], 4000) # U
            img = h5u.get_hwc_img(fn, event, im_tags, [1, 10], [800, 1600], [0, 600], 4000) # V

            print(img.shape)
            if img.shape[0] < img.shape[1]:
                print("Error: image height larger than the width")

            mask = predict_img(net=net,
                                full_img=img,
                                scale_factor=args.scale,
                                out_threshold=args.mask_threshold,
                                use_dense_crf= not args.no_crf,
                                use_gpu=not args.cpu)

            if args.viz:
                print("Visualizing results for image {}, close to continue ...".format(fn))
                h5u.plot_mask(mask)
                # h5u.plot_img(img)

            if not args.no_save:
                out_fn = out_files[i]
                result = mask_to_image(mask)
                result.save(out_files[i])

                print("Mask saved to {}".format(out_files[i]))
