import sys
import os
import math
import itertools
from optparse import OptionParser
import numpy as np
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from unet import UNet
from uresnet import UResNet
from nestedunet import NestedUNet
from hdf5_dataset import HDF5Dataset

from eval_util import eval_dice, eval_loss, eval_eff_pur
from utils import h5_utils as h5u

# Function to print the current learning rate
def print_lr(optimizer):
    for param_group in optimizer.param_groups:
        print(param_group['lr'])

# Function to apply exponential decay to the learning rate
def lr_exp_decay(optimizer, lr0, gamma, epoch):
    lr = lr0 * math.exp(-gamma * epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

# Main training function
def train_net(net,
              im_tags,
              ma_tags,
              truth_th,
              sepoch=0,  # Start epoch
              nepoch=1,  # Number of epochs to train
              strain=0,  # Start index for training samples
              ntrain=10,  # Number of training samples
              sval=450,  # Start index for validation samples
              nval=50,  # Number of validation samples
              batch_size=8,
              lr=0.1,  # Learning rate
              val_percent=0.10,  # Percentage of data to use for validation
              save_cp=True,  # Save checkpoints
              gpu=False,  # Use GPU if available
              img_scale=0.5,  # Image downscaling factor
              num_workers=0,  # Number of workers for data loading
              pin_memory=False,  # Use pinned memory for data loading
              drop_last=False,  # Drop the last incomplete batch
              prefetch_factor=2,  # Number of batches to prefetch
              persistent_workers=False,  # Keep data loading workers persistent
              dir_checkpoint=''):  # Directory to save checkpoints

    # Create the checkpoint directory if it doesn't exist
    if not os.path.exists(dir_checkpoint):
        os.makedirs(dir_checkpoint)
    
    # Create and shuffle the training and validation indices
    iddataset = {}
    iddataset['train'] = list(strain + np.arange(ntrain))
    np.random.shuffle(iddataset['train'])
    iddataset['val'] = list(sval + np.arange(nval))
    np.random.shuffle(iddataset['val'])
    
    # Create the training dataset
    train_dataset = HDF5Dataset(
        file_img=file_img,
        file_mask=file_mask,
        img_tags=im_tags,
        mask_tags=ma_tags,
        indices=iddataset['train'],
        rebin=[1, 10],
        x_range=[0, 1984],
        y_range=[0, 3500],
        z_scale=2000,
        truth_th=truth_th
    )
    
    # Create the validation dataset
    val_dataset = HDF5Dataset(
        file_img=file_img,
        file_mask=file_mask,
        img_tags=im_tags,
        mask_tags=ma_tags,
        indices=iddataset['val'],
        rebin=[1, 10],
        x_range=[0, 1984],
        y_range=[0, 3500],
        z_scale=2000,
        truth_th=truth_th
    )
    
    # Data loader arguments
    data_loader_args = {
        'batch_size': batch_size,
        'shuffle': True,  # Shuffle training data
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'drop_last': drop_last,
        'persistent_workers': persistent_workers
    }

    # Set prefetch factor if using multiple workers
    if num_workers > 0:
        data_loader_args['prefetch_factor'] = prefetch_factor
    
    # Create data loader for training data
    train_loader = DataLoader(train_dataset, **data_loader_args)
    
    # Modify to not shuffle validation data
    data_loader_args['shuffle'] = False
    
    # Create data loader for validation data
    val_loader = DataLoader(val_dataset, **data_loader_args)
    
    # Set up the optimizer and loss function
    # optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.BCELoss()

    best_val_dice = 0  # Initialize best validation dice coefficient
    best_val_loss = float('inf')  # Initialize best validation loss

    # Training loop
    for epoch in range(sepoch, sepoch + nepoch):
        net.train()  # Set the network to training mode
        epoch_loss = 0  # Initialize epoch loss

        # Iterate over the training data
        for imgs, true_masks in tqdm(train_loader):
            if gpu:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()

            masks_pred = net(imgs)  # Forward pass
            masks_probs_flat = masks_pred.view(-1)  # Flatten predictions
            true_masks_flat = true_masks.view(-1)  # Flatten true masks

            loss = criterion(masks_probs_flat, true_masks_flat)  # Compute loss
            epoch_loss += loss.item()

            optimizer.zero_grad()  # Zero the gradients
            loss.backward()  # Backward pass
            optimizer.step()  # Update the parameters

        # Compute average epoch loss
        epoch_loss = epoch_loss / len(train_loader)
        print('Epoch finished! Loss: {:.6f}'.format(epoch_loss))

        net.eval()  # Set the network to evaluation mode
        with torch.no_grad():  # Disable gradient computation
            val_dice = eval_dice(net, val_loader, gpu)  # Evaluate dice coefficient
            val_loss = eval_loss(net, criterion, val_loader, gpu)  # Evaluate loss
            print('Validation Dice Coeff: {:.4f}, Validation Loss: {:.6f}'.format(val_dice, val_loss))

            # Save the best model based on dice coefficient
            if val_dice > best_val_dice:
                best_val_dice = val_dice
                torch.save(net.state_dict(), dir_checkpoint + 'best_dice.pth')
                print('New best dice model saved!')

            # Save the best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(net.state_dict(), dir_checkpoint + 'best_loss.pth')
                print('New best loss model saved!')

        # Save CSV files for loss and dice coefficient
        with open(dir_checkpoint + 'loss.csv', 'a') as f_loss, open(dir_checkpoint + 'dice.csv', 'a') as f_dice:
            f_loss.write(f'{epoch},{epoch_loss},{val_loss}\n')
            f_dice.write(f'{epoch},{val_dice}\n')

# Function to parse command-line arguments
def get_args():
    parser = OptionParser()
    parser.add_option('--start-epoch', dest='sepoch', default=0, type='int',
                      help='start epoch number')
    parser.add_option('-e', '--nepoch', dest='nepoch', default=1, type='int',
                      help='number of epochs')

    parser.add_option('--start-train', dest='strain', default=0, type='int',
                      help='start sample for training')
    parser.add_option('--ntrain', dest='ntrain', default=10, type='int',
                      help='number of sample for training')
    parser.add_option('--start-val', dest='sval', default=450, type='int',
                      help='start sample for val')
    parser.add_option('--nval', dest='nval', default=50, type='int',
                      help='number of sample for nval')

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
    
    # Add DataLoader parameters
    parser.add_option('--num-workers', dest='num_workers', default=0, type='int',
                      help='number of data loading workers')
    parser.add_option('--pin-memory', action='store_true', dest='pin_memory',
                      default=False, help='use pin memory')
    parser.add_option('--drop-last', action='store_true', dest='drop_last',
                      default=False, help='drop last incomplete batch')
    parser.add_option('--prefetch-factor', dest='prefetch_factor', default=2, type='int',
                      help='number of batches to prefetch')
    parser.add_option('--persistent-workers', action='store_true', dest='persistent_workers',
                      default=False, help='keep data loading workers persistent')
    
    (options, args) = parser.parse_args()
    return options

# Function to extract the base name for the directory from the file path
def extract_base_name(file_path):
    return os.path.basename(file_path).split('-')[0]

if __name__ == '__main__':
    args = get_args()  # Parse command-line arguments

    torch.set_num_threads(1)  # Set the number of threads for PyTorch

    im_tags = ['frame_loose_lf1', 'frame_mp2_roi1', 'frame_mp3_roi1']  # Image tags
    ma_tags = ['frame_deposplat1']  # Mask tags
    truth_th = 100  # Truth threshold

    # File paths for the image and mask files
    file_img = '/scratch/7DayLifetime/munjung/DNN_ROI/train/smeared/tpc1_bothplanes_with_prolongedtrks-rec.h5'
    file_mask = '/scratch/7DayLifetime/munjung/DNN_ROI/train/smeared/tpc1_bothplanes_with_prolongedtrks-tru.h5'

    # Extract the base name for the directory
    base_name = extract_base_name(file_img)
    dir_checkpoint = f'/home/abhat/wirecell_sbnd/Pytorch-UNet/checkpoints/UResNet/{base_name}/'

    # Initialize the network
    # net = UNet(len(im_tags), len(ma_tags))
    net = UResNet(len(im_tags), len(ma_tags))
    # net = NestedUNet(len(im_tags), len(ma_tags))

    # Load a pre-trained model if specified
    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    # Move the network to GPU if available
    if args.gpu:
        net.cuda()

    try:
        # Train the network
        train_net(net=net,
                  im_tags=im_tags,
                  ma_tags=ma_tags,
                  truth_th=truth_th,
                  sepoch=args.sepoch,
                  nepoch=args.nepoch,
                  strain=args.strain,
                  ntrain=args.ntrain,
                  sval=args.sval,
                  nval=args.nval,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  gpu=args.gpu,
                  img_scale=args.scale,
                  num_workers=args.num_workers,
                  pin_memory=args.pin_memory,
                  drop_last=args.drop_last,
                  prefetch_factor=args.prefetch_factor,
                  persistent_workers=args.persistent_workers,
                  dir_checkpoint=dir_checkpoint)
    except KeyboardInterrupt:
        # Save the model if training is interrupted
        torch.save(net.state_dict(), dir_checkpoint + 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
