#!/usr/bin/env python

import datetime
import sys
import os
import itertools
from optparse import OptionParser
from tqdm import tqdm
import json

import math
import numpy as np
import pandas as pd

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from unet import UNet
from uresnet import UResNet
from nestedunet import NestedUNet

from eval_util import eval_dice_loss, eval_eff_pur
from dice_loss import dice_coeff
from utils import get_ids, split_ids, split_train_val, get_imgs_and_masks, batch
from utils import h5_utils as h5u
from utils import log_fig, ep_fig
from hdf5_dataset import HDF5Dataset


def eval_img(net, dataset, gpu=False):
    for i, b in enumerate(dataset):
        img       = b[0]
        true_mask = b[1]
        img       = torch.from_numpy(img).unsqueeze(0)
        true_mask = torch.from_numpy(true_mask).unsqueeze(0)

        if gpu:
            img       = img.cuda()
            true_mask = true_mask.cuda()

        mask_pred = net(img)[0]

        return true_mask.cpu().numpy(), mask_pred.cpu().numpy()

def print_lr(optimizer):
    for param_group in optimizer.param_groups:
        print(param_group['lr'])


def lr_exp_decay(optimizer, lr0, gamma, epoch):
    lr = lr0*math.exp(-gamma*epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def train_net(net,
              gpu            = False,
              save_cp        = False,
              dir_checkpoint = "",
              batch_size     = 2,
              lr             = 0.1,
              train_dir      = "",
              ntrainfiles   = 100,
              sepoch         = 0,
              nepoch         = 1,
              strain         = 0,
              ntrain         = 10,
              sval           = 10,
              nval           = 1,
              img_scale      = [1,10],
              x_range        = [0,1984],
              y_range        = [0,3500],
              z_scale        = 2000,
              truth_th       = 100,
              im_tags        = ['frame_loose_lf1', 'frame_mp2_roi1', 'frame_mp3_roi1'],
              ma_tags        = ['frame_deposplat1']):

    # log file
    if not os.path.exists(dir_checkpoint):
        os.mkdir(dir_checkpoint)
    outfile_log        = open(dir_checkpoint+'/log', 'a+')
    outfile_loss_batch = open(dir_checkpoint+'/train-loss-batch.csv','a+')
    outfile_dice_batch = open(dir_checkpoint+'/train-dice-batch.csv','a+')
    outfile_loss       = open(dir_checkpoint+'/train-loss.csv','a+')
    outfile_dice       = open(dir_checkpoint+'/train-dice.csv','a+')
    outfile_eval_loss  = open(dir_checkpoint+'/eval-loss.csv','a+')
    outfile_eval_dice  = open(dir_checkpoint+'/eval-dice.csv','a+')
    outfile_ep         = open(dir_checkpoint+'/ep.csv','a+')

    DT_STR = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(dir_checkpoint+"/tensorboard/"+DT_STR)

    print('''
    im_tags: {}
    ma_tags: {}
    truth_th: {}
    '''.format(im_tags,
               ma_tags,
               truth_th), 
          file=outfile_log, flush=True)

    files_samples    = os.listdir(train_dir)
    files_samples    = [f for f in files_samples if 'tpc0' in f]
    files_img        = [f for f in files_samples if 'rec' in f][:ntrainfiles]
    files_mask       = [f.replace("rec","tru") for f in files_img]
    files_img         = [os.path.join(train_dir, f) for f in files_img]
    files_mask        = [os.path.join(train_dir, f) for f in files_mask]

    print('''
    files_img: {}
    files_mask: {}
    '''.format(files_img, 
               files_mask), file=outfile_log, flush=True)

    iddataset = {}
    iddataset['train'] = list(1+strain+np.arange(ntrain))
    iddataset['val']   = list(1+sval+np.arange(nval))
    np.random.shuffle(iddataset['train'])
    np.random.shuffle(iddataset['val'])
    N_train = len(iddataset['train'])
    N_val   = len(iddataset['val'])
    print(iddataset['train'], file=outfile_log, flush=True)
    print(iddataset['val'], file=outfile_log, flush=True)
    print("iddataset['train'] len = ", N_train)
    print("iddataset['val'] len = ", N_val)

    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Image scale: {}
        X range: {}
        Y range: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(nepoch, 
               batch_size, 
               lr, 
               N_train,
               N_val, 
               str(img_scale),
               str(x_range),
               str(y_range),
               str(save_cp), 
               str(gpu)), 
          file=outfile_log, flush=True)

    # configure dataloader
    num_workers=0  # Number of workers for data loading
    pin_memory=False  # Use pinned memory for data loading
    drop_last=False  # Drop the last incomplete batch
    prefetch_factor=2  # Number of batches to prefetch
    persistent_workers=False  # Keep data loading workers persistent
    
    data_loader_args = {
        'batch_size': batch_size,
        'shuffle': True,  # Shuffle training data
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'drop_last': drop_last,
        'persistent_workers': persistent_workers
    }
    if num_workers > 0:
        data_loader_args['prefetch_factor'] = prefetch_factor

    # train
    if sepoch > 0 :
        net.load_state_dict(torch.load('{}/CP{}.pth'.format(dir_checkpoint, sepoch-1)))
    
    optimizer = optim.SGD(net.parameters(), 
                          lr=lr, 
                          momentum=0.9, 
                          weight_decay=0.0005)
    # optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.BCELoss()

    n_update_train = 0
    n_update_val = 0
    best_val_dice = 0
    best_val_loss = np.inf
    for epoch in range(sepoch, sepoch+nepoch):
        print('Starting epoch {}/{}.'.format(epoch, nepoch))

        # load data
        train_dataset = HDF5Dataset(
            files_img=files_img,
            files_mask=files_mask,
            img_tags=im_tags,
            mask_tags=ma_tags,
            indices=iddataset['train'],
            rebin=img_scale,
            x_range=x_range,
            y_range=y_range,
            z_scale=z_scale,
            truth_th=truth_th
        )
        train_loader = DataLoader(train_dataset, **data_loader_args)

        val_dataset = HDF5Dataset(
            files_img=files_img,
            files_mask=files_mask,
            img_tags=im_tags,
            mask_tags=ma_tags,
            indices=iddataset['val'],
            rebin=img_scale,
            x_range=x_range,
            y_range=y_range,
            z_scale=z_scale,
            truth_th=truth_th
        )
        val_loader = DataLoader(val_dataset, **data_loader_args)

        # train

        scheduler = optimizer
        # scheduler = lr_exp_decay(optimizer, lr, 0.04, epoch)
        print(scheduler, file=outfile_log, flush=True)

        net.train()
        epoch_loss = 0
        epoch_dice = 0
        for imgs, true_masks in tqdm(train_loader):
            if gpu:
                imgs       = imgs.cuda()
                true_masks = true_masks.cuda()

            masks_pred       = net(imgs)
            masks_probs_flat = masks_pred.view(-1)
            true_masks_flat  = true_masks.view(-1)

            loss = criterion(masks_probs_flat, true_masks_flat)
            epoch_loss += loss.item()

            scheduler.zero_grad()
            loss.backward()
            scheduler.step()

        epoch_loss = epoch_loss / len(train_loader)
        print('Epoch finished ! Loss: {:.6f}'.format(epoch_loss))
        print('{:.4f}, {:.6f}'.format(epoch, epoch_loss), file=outfile_loss, flush=True)
        writer.add_scalar('loss/train', epoch_loss, epoch)

        if save_cp:
            torch.save(net.state_dict(), dir_checkpoint + 'CP{}.pth'.format(epoch))
            print('Checkpoint e{} saved !'.format(epoch))

        net.eval()
        with torch.no_grad():

            # validation
            # val1, val2, val3 = itertools.tee(val_loader, 3)

            val_dice, val_loss = eval_dice_loss(net, val_loader, criterion, gpu)
            if val_dice > best_val_dice:
                best_val_dice = val_dice
                torch.save(net.state_dict(), dir_checkpoint + '/best_dice.pth')
                print('********* New Best Dice Coeff: {:.4f}'.format(best_val_dice))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(net.state_dict(), dir_checkpoint + '/best_loss.pth')
                print('********* New Best Loss: {:.4f}'.format(best_val_loss))

            print('Validation Dice Coeff: {:.4f}, {:.6f}'.format(epoch, val_dice))
            print('{:.4f}, {:.6f}'.format(epoch, val_dice), file=outfile_eval_dice, flush=True)
            print('Validation Loss: {:.4f}, {:.6f}'.format(epoch, val_loss))
            print('{:.4f}, {:.6f}'.format(epoch, val_loss), file=outfile_eval_loss, flush=True)
            writer.add_scalar('loss/validation', val_loss, epoch)
            writer.add_scalar('dice/validation', val_dice, epoch)


def read_config(cfgname):
    config = None
    with open(cfgname, 'r') as fin:
        config = json.loads(fin.read());
    if config is None:
        print ('This script requires configuration file: config.json')
        exit(1)
    return config


def get_args():
    parser = OptionParser()
    parser.add_option('-c', '--config', 
                      help='JSON with script configuration')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu', default=False, 
                      help='use cuda')
    parser.add_option('-s', '--savecp', dest='savecp', default=False, 
                      help='save checkpoints')
    parser.add_option('-l', '--load', dest='load', default=False, 
                      help='load file model')
    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':
    args = get_args()
    config = read_config(args.config)
    torch.set_num_threads(1)

    net = UNet(len(config["im_tags"]), len(config["ma_tags"]))
    # net = UResNet(len(config["im_tags"]), len(config["ma_tags"]))
    # net = NestedUNet(len(config["im_tags"]),len(config["ma_tags"]))

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    if args.gpu:
        net.cuda()
        # cudnn.benchmark = True # faster convolutions, but more memory

    try:
        train_net(net            = net,
                  gpu            = args.gpu,
                  save_cp        = args.savecp,
                  dir_checkpoint = config["dir_checkpoint"],
                  batch_size     = config["batch_size"],
                  lr             = config["learning_rate"],
                  train_dir      = config["train_dir"],
                  ntrainfiles    = config["ntrainfiles"],
                  sepoch         = config["start_epoch"],
                  nepoch         = config["nepoch"],
                  strain         = config["start_train"],
                  ntrain         = config["ntrain"],
                  sval           = config["start_val"],
                  nval           = config["nval"],
                  img_scale      = config["scale"],
                  x_range        = config["x_range"],
                  y_range        = config["y_range"],
                  z_scale        = config["z_scale"],
                  truth_th       = config["truth_th"],
                  im_tags        = config["im_tags"],
                  ma_tags        = config["ma_tags"],
                  )
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
