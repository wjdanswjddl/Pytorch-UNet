#!/usr/bin/env python

import sys
import os
from optparse import OptionParser
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim

from unet import UNet
from uresnet import UResNet
from nestedunet import NestedUNet

from eval import eval_net
from utils import get_ids, split_ids, split_train_val, get_imgs_and_masks, batch
from utils import h5_utils as h5u

def train_net(net,
              im_tags = ['frame_loose_lf0', 'frame_mp2_roi0', 'frame_mp3_roi0'],
              ma_tags = ['frame_ductor0'],
              truth_th = 100,
              epochs=5,
              samples=10,
              batch_size=10,
              lr=0.1,
              val_percent=0.10,
              save_cp=True,
              gpu=False,
              img_scale=0.5):

    dir_checkpoint = 'checkpoints/'

    ids = list(np.arange(samples))

    iddataset = split_train_val(ids, val_percent)
    print(iddataset['train'])

    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr, len(iddataset['train']),
               len(iddataset['val']), str(save_cp), str(gpu)))

    N_train = len(iddataset['train'])

    optimizer = optim.SGD(net.parameters(),
                          lr=lr,
                          momentum=0.9,
                          weight_decay=0.0005)

    criterion = nn.BCELoss()

    print('''
    im_tags: {}
    ma_tags: {}
    truth_th: {}
    '''.format(im_tags,ma_tags,truth_th))
    outfile_loss = open('loss.csv','w')
    for epoch in range(0,epochs):

        file_img  = 'data/cosmic-rec-0.h5'
        file_mask = 'data/cosmic-tru-0.h5'
        # if epoch % 2 != 0 :
        #   file_img  = 'data/mu-rec-0.h5'
        #   file_mask = 'data/mu-tru-0.h5'

        print('''
        file_img: {}
        file_mask: {}
        '''.format(file_img, file_mask))

        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        net.train()

        train = zip(
          h5u.get_chw_imgs(file_img, iddataset['train'], im_tags, [1, 10], [800, 1600], [0, 600], 4000),
          h5u.get_masks(file_mask,   iddataset['train'], ma_tags, [1, 10], [800, 1600], [0, 600], truth_th)
        )
        val = zip(
          h5u.get_chw_imgs(file_img, iddataset['val'],   im_tags, [1, 10], [800, 1600], [0, 600], 4000),
          h5u.get_masks(file_mask,   iddataset['val'],   ma_tags, [1, 10], [800, 1600], [0, 600], truth_th)
        )

        # for img, mask in train:
        #   print(img.shape)
        #   print(mask.shape)
        #   h5u.plot_and_mask(np.transpose(img, axes=[1, 2, 0]), mask)
        # continue

        epoch_loss = 0

        for i, b in enumerate(batch(train, batch_size)):
            imgs = np.array([i[0] for i in b]).astype(np.float32)
            true_masks = np.array([i[1] for i in b])

            # print(imgs.shape)
            # print(true_masks.shape)
            # continue

            imgs = torch.from_numpy(imgs)
            true_masks = torch.from_numpy(true_masks)

            if gpu:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()

            masks_pred = net(imgs)
            # print("Truth: ", np.count_nonzero(true_masks.cpu().detach().numpy()))
            # print("Pred:  ", np.count_nonzero(masks_pred.cpu().detach().numpy()>0.5))
            masks_probs_flat = masks_pred.view(-1)

            true_masks_flat = true_masks.view(-1)

            loss = criterion(masks_probs_flat, true_masks_flat)
            epoch_loss += loss.item()

            print('{} : {:.4f} --- loss: {:.6f}'.format(epoch, i * batch_size / N_train, loss.item()))
            print('{:.4f}, {:.6f}'.format(i * batch_size / N_train, loss.item()), file=outfile_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if save_cp and i%20==0:
            #     torch.save(net.state_dict(),
            #               dir_checkpoint + 'CP{}-{}.pth'.format(epoch + 1,i+1))
            #     print('Checkpoint e{}b{} saved !'.format(epoch + 1,i+1))

        print('Epoch finished ! Loss: {:.6f}'.format(epoch_loss / i))

        if save_cp:
            torch.save(net.state_dict(),
                      dir_checkpoint + 'CP{}-{}.pth'.format(epoch + 1,i+1))
            print('Checkpoint e{} saved !'.format(epoch + 1))

        if False:
            val_dice = eval_net(net, val, gpu)
            print('Validation Dice Coeff: {}'.format(val_dice))




def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=1, type='int',
                      help='number of epochs')
    parser.add_option('-n', '--samples', dest='samples', default=10, type='int',
                      help='number of samples')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=1,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.1,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=False, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=0.5, help='downscaling factor of the images')

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()

    torch.set_num_threads(1)

    # im_tags = ['frame_tight_lf0', 'frame_loose_lf0'] #lt
    # im_tags = ['frame_loose_lf0', 'frame_mp2_roi0', 'frame_mp3_roi0']    # l23
    im_tags = ['frame_loose_lf0', 'frame_tight_lf0', 'frame_mp2_roi0', 'frame_mp3_roi0']    # lt23
    ma_tags = ['frame_ductor0']
    truth_th = 100

    net = UNet(len(im_tags), len(ma_tags))
    # net = UResNet(len(im_tags), len(ma_tags))
    # net = NestedUNet(len(im_tags),len(ma_tags))

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    if args.gpu:
        net.cuda()
        # cudnn.benchmark = True # faster convolutions, but more memory

    try:
        train_net(net=net,
                  im_tags=im_tags,
                  ma_tags=ma_tags,
                  truth_th=truth_th,
                  epochs=args.epochs,
                  samples=args.samples,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  gpu=args.gpu,
                  img_scale=args.scale)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
