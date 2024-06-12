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

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from unet import UNet
from uresnet import UResNet
from nestedunet import NestedUNet

from eval_util import eval_dice, eval_loss, eval_eff_pur
from dice_loss import dice_coeff
from utils import get_ids, split_ids, split_train_val, get_imgs_and_masks, batch
from utils import h5_utils as h5u
import matplotlib.pyplot as plt

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

def log_fig(true_img, pred_img, vval=1):
    fig, ax = plt.subplots(figsize=(8, 5))
    true_fig = plt.imshow(true_img.T, 
                          aspect="auto", origin="lower", vmin=-vval, vmax=vval, cmap="bwr")
    plt.xlabel("wire")
    plt.ylabel("tick")
    plt.title("True")
    plt.colorbar()

    fig, ax = plt.subplots(figsize=(8, 5))
    pred_fig = plt.imshow(pred_img.T, 
                          aspect="auto", origin="lower", vmin=-vval, vmax=vval, cmap="bwr")
    plt.xlabel("wire")
    plt.ylabel("tick")
    plt.title("Prediction")
    plt.colorbar()

    return true_fig.get_figure(), pred_fig.get_figure()


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
              sample         = "",
              target         = "",
              test_dir       = "",
              test_tags      = ["75-80"],
              sepoch         = 0,
              nepoch         = 1,
              strain         = 0,
              ntrain         = 10,
              sval           = 10,
              nval           = 1,
              stest          = 0,
              ntest          = 1,
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

    file_img       = sample
    file_mask      = target
    file_test_img  = test_dir+"/tpc0_plane0_85-87_1000-rec.h5"
    file_test_mask = test_dir+"/tpc0_plane0_85-87_1000-tru.h5"

    print('''
    file_img: {}
    file_mask: {}
    '''.format(file_img, 
               file_mask), file=outfile_log, flush=True)

    iddataset = {}
    iddataset['train'] = list(strain+np.arange(ntrain))
    iddataset['val']   = list(sval+np.arange(nval))
    iddataset['test']   = list(stest+np.arange(ntest))
    np.random.shuffle(iddataset['train'])
    np.random.shuffle(iddataset['val'])
    np.random.shuffle(iddataset['test'])
    N_train = len(iddataset['train'])
    N_val   = len(iddataset['val'])
    N_test   = len(iddataset['test'])
    print(iddataset['train'], file=outfile_log, flush=True)
    print(iddataset['val'], file=outfile_log, flush=True)
    print(iddataset['test'], file=outfile_log, flush=True)
    print("iddataset['train'] len = ", N_train)
    print("iddataset['val'] len = ", N_val)
    print("iddataset['test'] len = ", N_test)

    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Test size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(nepoch, 
               batch_size, 
               lr, 
               N_train,
               N_val, 
               N_test,
               str(save_cp), 
               str(gpu)), 
          file=outfile_log, flush=True)

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
        train = zip(
          h5u.get_chw_imgs(file_img, iddataset['train'], im_tags, img_scale, x_range, y_range, z_scale),
          h5u.get_masks(file_mask,   iddataset['train'], ma_tags, img_scale, x_range, y_range, truth_th)
        )
        val = zip(
          h5u.get_chw_imgs(file_img, iddataset['val'], im_tags, img_scale, x_range, y_range, z_scale),
          h5u.get_masks(file_mask,   iddataset['val'], ma_tags, img_scale, x_range, y_range, truth_th)
        )
        test = zip(
          h5u.get_chw_imgs(file_test_img, iddataset['test'], im_tags, img_scale, x_range, y_range, z_scale),
          h5u.get_masks(file_test_mask,   iddataset['test'], ma_tags, img_scale, x_range, y_range, truth_th)
        )

        # train
        scheduler = optimizer
        # scheduler = lr_exp_decay(optimizer, lr, 0.04, epoch)
        print(scheduler, file=outfile_log, flush=True)

        net.train()
        epoch_loss = 0
        epoch_dice = 0
        for i, b in tqdm(enumerate(batch(train, batch_size)), total=N_train//batch_size+1, desc='Training'):
            #TODO: float32 is unnecessary
            imgs       = np.array([i[0] for i in b]).astype(np.float32)
            imgs       = torch.from_numpy(imgs)
            true_masks = np.array([i[1] for i in b])
            true_masks = torch.from_numpy(true_masks)

            if gpu:
                imgs       = imgs.cuda()
                true_masks = true_masks.cuda()

            masks_pred       = net(imgs)
            masks_probs_flat = masks_pred.view(-1)
            true_masks_flat  = true_masks.view(-1)

            loss = criterion(masks_probs_flat, true_masks_flat)
            print('{:.4f}, {:.6f}'.format(epoch + i * batch_size / N_train, loss.item()), file=outfile_loss_batch, flush=True)
            epoch_loss += loss.item()

            scheduler.zero_grad()
            loss.backward()
            scheduler.step()

        epoch_loss = epoch_loss / (i + 1)
        print('Epoch finished ! Loss: {:.6f}'.format(epoch_loss))
        print('{:.4f}, {:.6f}'.format(epoch, epoch_loss), file=outfile_loss, flush=True)
        writer.add_scalar('loss/train', epoch_loss, epoch)

        if save_cp:
            torch.save(net.state_dict(), dir_checkpoint + 'CP{}.pth'.format(epoch))
            print('Checkpoint e{} saved !'.format(epoch))

        # validation
        net.eval()
        with torch.no_grad():
            val1, val2, val3 = itertools.tee(val, 3)

            val_dice = eval_dice(net, val1, gpu)
            if val_dice > best_val_dice:
                best_val_dice = val_dice
                torch.save(net.state_dict(), dir_checkpoint + '/best_dice.pth')
                print('********* New Best Dice Coeff: {:.4f}'.format(best_val_dice))

            val_loss = eval_loss(net, criterion, val2, gpu)
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

            true_img, pred_img = eval_img(net, val3, gpu)
            val_true_img, val_pred_img = log_fig(true_img, pred_img)
            writer.add_figure("val/true", val_true_img, epoch)
            writer.add_figure("val/pred", val_pred_img, epoch)

            #TODO: evaluation on test samples
            test1, test2, test3 = itertools.tee(test, 3)
            # pixel, roi evaluation
            ep = eval_eff_pur(net, test1, 0.5, args.gpu)
            print('{}, {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format("test", ep[0], ep[1], ep[2], ep[3]))
            print('{}, {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format("test", ep[0], ep[1], ep[2], ep[3]), file=outfile_ep)
            writer.add_scalar('test/pixel_eff', ep[0], epoch)
            writer.add_scalar('test/pixel_pur', ep[1], epoch)

            true_img, pred_img = eval_img(net, test3, gpu)
            test_true_img, test_pred_img = log_fig(true_img, pred_img)
            writer.add_figure("test/true", test_true_img, epoch)
            writer.add_figure("test/pred", test_pred_img, epoch)


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
                  sample         = config["sample"],
                  target         = config["target"],
                  test_dir       = config["test_dir"],
                  test_tags      = config["test_tags"],
                  sepoch         = config["start_epoch"],
                  nepoch         = config["nepoch"],
                  strain         = config["start_train"],
                  ntrain         = config["ntrain"],
                  sval           = config["start_val"],
                  nval           = config["nval"],
                  stest          = config["start_test"],
                  ntest          = config["ntest"],
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
