#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import sys

inputs   = [
    # train-eval
    # 'model/unet-l23-cosmic500-e50-t1/loss.csv',
    # 'model/unet-l23-cosmic500-e50-t1/eval-loss.csv',
    
    # 'model/unet-explr-l23-cosmic500-e100/loss.csv',
    # 'model/unet-explr-l23-cosmic500-e100/eval-loss.csv',

    # 'model/unet-adam-l23-cosmic500-e50/loss.csv',
    # 'model/unet-adam-l23-cosmic500-e50/eval-loss.csv',
    
    'test3-th10/loss.csv',
    'test3-th10/eval-loss.csv',

    # sample size
    # 'model/unet-l23-cosmic500-e50-t1/ep-87-85.csv',
    # 'sample-size-test-400/ep-87-85.csv',
    # 'sample-size-test-300/ep-87-85.csv',
    # 'sample-size-test-200/ep-87-85.csv',
    # 'sample-size-test-50/ep-87-85.csv',
    ]
labels = [
    # train-eval
    'Training',
    'Validation',

    # sample size
    # '450',
    # '400',
    # '300',
    # '200',
    # '50',
    ]

for itag in range(len(inputs)) :
    data = np.genfromtxt(inputs[itag], delimiter=',')
    marker = '-o'
    if labels[itag].find('Val') > 0:
        marker = '-^'
    plt.plot(data[:,0], data[:,int(sys.argv[1])], marker,label=labels[itag])

fontsize = 26
plt.legend(loc='best',fontsize=fontsize)
plt.grid()
# plt.ylim(0.003,0.010)
# plt.yscale('log')
plt.xlabel("Epoch", fontsize=fontsize)
plt.ylabel("Mean Loss", fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
# plt.ylabel("Pixel Efficiency", fontsize=fontsize)
# plt.ylabel("Pixel Purity", fontsize=fontsize)
# plt.ylabel("ROI Efficiency", fontsize=fontsize)
# plt.ylabel("ROI Purity", fontsize=fontsize)
plt.show()