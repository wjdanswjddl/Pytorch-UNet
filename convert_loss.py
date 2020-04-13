#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

tags   = [
    'model/unet-l23-cosmic500-e50',
    'model/uresnet-l23-cosmic500-e50',
    'model/nestedunet-l23-cosmic500-e50',
    # 'uresnet-explr-l23-cosmic500',
    # 'uresnet-l23-cosmic500-t1',
    ]
labels = [
    'UNet',
    'UResNet',
    'NestedUNet',
    # 'UResNet-ExpLR',
    # 'UResNet-t1',
    ]
dfs = [pd.read_csv(tag+'/loss.csv', sep=' ', header=None) for tag in tags]

epoch = 50
nsample = 450

epoch_index = np.linspace(0, epoch-1, num=epoch)
mean_loss = np.zeros((len(tags),epoch))

for itag in range(len(tags)) :
    for iepoch in range(epoch):
        mean_loss[itag][iepoch] = dfs[itag][1][nsample*iepoch:nsample*(iepoch+1)].to_numpy().mean()
    out = np.concatenate(( np.expand_dims(epoch_index,axis=1), np.expand_dims(mean_loss[itag],axis=1) ), axis=1)
    np.savetxt(labels[itag]+'.csv',out,delimiter=',')