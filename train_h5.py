#!/usr/bin/env python

import sys
import os
from optparse import OptionParser
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim

from eval import eval_net
from unet import UNet
from utils import get_ids, split_ids, split_train_val, get_imgs_and_masks, batch
from utils import h5_utils as h5u

def train_net(net,
              epochs=5,
              batch_size=10,
              lr=0.1,
              val_percent=0.05,
              save_cp=True,
              gpu=False,
              img_scale=0.5):

    dir_img = 'data/train/'
    dir_mask = 'data/train_masks/'
    dir_checkpoint = 'checkpoints/'

    file_img  = 'data/g4-rec-0.h5'
    file_mask = 'data/g4-tru-0.h5'

    ids = list(np.arange(100))

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

    # reset the generators
    im_tags = ['frame_tight_lf0', 'frame_loose_lf0', 'frame_gauss0']
    im_tags = ['frame_tight_lf0', 'frame_tight_lf0', 'frame_loose_lf0']
    im_tags = ['frame_tight_lf0', 'frame_loose_lf0', 'frame_loose_lf0']
    ma_tags = ['frame_ductor0']
    truth_th = 50

    print('''
    file_img: {}
    file_mask: {}
    im_tags: {}
    ma_tags: {}
    truth_th: {}
    '''.format(file_img, file_mask, im_tags,ma_tags,truth_th))

    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        net.train()

        train = zip(
          h5u.get_chw_imgs(file_img, iddataset['train'], im_tags, [1, 10], [0, 800], [0, 600], 4000),
          h5u.get_masks(file_mask, iddataset['train'], ma_tags, [1, 10], [0, 800], [0, 600], truth_th)
        )
        val = zip(
          h5u.get_chw_imgs(file_img, iddataset['val'], im_tags, [1, 10], [0, 800], [0, 600], 4000),
          h5u.get_masks(file_mask, iddataset['val'], ma_tags, [1, 10], [0, 800], [0, 600], truth_th)
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

            print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_size / N_train, loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if save_cp and i%20==0:
                torch.save(net.state_dict(),
                          dir_checkpoint + 'CP{}-{}.pth'.format(epoch + 1,i+1))
                print('Checkpoint e{}b{} saved !'.format(epoch + 1,i+1))

        print('Epoch finished ! Loss: {}'.format(epoch_loss / i))

        if save_cp:
            torch.save(net.state_dict(),
                      dir_checkpoint + 'CP{}-{}.pth'.format(epoch + 1,i+1))
            print('Checkpoint e{} saved !'.format(epoch + 1))

        if False:
            val_dice = eval_net(net, val, gpu)
            print('Validation Dice Coeff: {}'.format(val_dice))




def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=10,
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

    net = UNet(n_channels=3, n_classes=1)

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    if args.gpu:
        net.cuda()
        # cudnn.benchmark = True # faster convolutions, but more memory

    try:
        train_net(net=net,
                  epochs=args.epochs,
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
