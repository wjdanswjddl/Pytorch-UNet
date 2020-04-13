#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

inputs   = [
    # 'model/unet-l23-cosmic500-e50/loss.csv',
    # 'unet-l23-cosmic500-e50-t1/loss.csv',
    # 'unet-l23-cosmic500-e50-t2/loss.csv',
    
    # 'unet-l23-cosmic500-e50-t1/eval-loss.csv',
    # 'unet-l23-cosmic500-e50-t2/eval-loss.csv',
    
    'unet-l23-cosmic500-e50-t1/ep-87-85.csv',
    'unet-l23-cosmic500-e50-t2/ep-87-85.csv',
    'unet-adam-l23-cosmic500-e50/ep-87-85.csv',
    ]
labels = [
    # 'UNet-LR0.1-e50-t0',
    # 'UNet-LR0.1-e50-t1',
    # 'UNet-LR0.1-e50-t2',

    # 'UNet-LR0.1-e50-t1-Val',
    # 'UNet-LR0.1-e50-t2-Val',
    
    'UNet-LR0.1-e50-t1-87-85',
    'UNet-LR0.1-e50-t2-87-85',
    'UNet-Adam-LR0.1-e50-t2-87-85',
    ]

for itag in range(len(inputs)) :
    data = np.genfromtxt(inputs[itag], delimiter=',')
    marker = '-o'
    if labels[itag].find('Val') > 0:
        marker = '-^'
    plt.plot(data[:,0], data[:,3], marker,label=labels[itag])

fontsize = 18
plt.legend(loc='best',fontsize=fontsize)
plt.grid()
# plt.ylim(0.003,0.010)
plt.xlabel("Epoch", fontsize=fontsize)
# plt.ylabel("Mean Loss", fontsize=fontsize)
plt.ylabel("Pixel Efficiency", fontsize=fontsize)
plt.ylabel("Pixel Purity", fontsize=fontsize)
plt.ylabel("ROI Efficiency", fontsize=fontsize)
# plt.ylabel("ROI Purity", fontsize=fontsize)
plt.show()