#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

tags   = ['model/unet-l23-cosmic500-e50', 'model/uresnet-512-l23-cosmic500-e50', 'model/nestedunet-l23-cosmic500-e50']
labels = ['UNet', 'UResNet', 'NestedUNet']
dfs = [pd.read_csv(tag+'/loss.csv', sep=' ', header=None) for tag in tags]

epoch = 50
nsample = 450

mean_loss = np.zeros((len(tags),epoch))

for itag in range(len(tags)) :
  for iepoch in range(epoch):
    mean_loss[itag][iepoch] = dfs[itag][1][nsample*iepoch:nsample*(iepoch+1)].to_numpy().mean()

for itag in range(len(tags)) :
  plt.plot(np.linspace(2, epoch,epoch-1), mean_loss[itag][1:],'o',label=labels[itag])
plt.legend(loc='best',fontsize=15)
plt.grid()
# plt.ylim(0,1e-2)
plt.xlabel("epoch", fontsize=15)
plt.ylabel("mean loss", fontsize=15)
plt.show()