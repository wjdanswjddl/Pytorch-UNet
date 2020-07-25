#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

inputs   = [
    # 'model/unet-explr-l23-cosmic500-e100/loss.csv',
    # 'model/unet-explr-l23-cosmic500-e100/eval-loss.csv',
    
    'loss-diff-3/loss.csv',
    'loss-diff-3/eval-loss.csv',

    # 'checkpoints/loss.csv',
    # 'checkpoints/eval-loss.csv',
    ]
labels = [
    'Training',
    'Validation',

    # '87-85',
    ]

for itag in range(len(inputs)) :
    data = np.genfromtxt(inputs[itag], delimiter=',')
    marker = '-o'
    if labels[itag].find('Val') > 0:
        marker = '-^'
    plt.plot(data[0:4,0], data[0:4,1], marker,label=labels[itag])

fontsize = 18
plt.legend(loc='best',fontsize=fontsize)
plt.grid()
# plt.ylim(0.003,0.010)
plt.xlabel("Epoch", fontsize=fontsize)
plt.ylabel("Mean Loss", fontsize=fontsize)
# plt.ylabel("Pixel Efficiency", fontsize=fontsize)
# plt.ylabel("Pixel Purity", fontsize=fontsize)
# plt.ylabel("ROI Efficiency", fontsize=fontsize)
# plt.ylabel("ROI Purity", fontsize=fontsize)
plt.show()