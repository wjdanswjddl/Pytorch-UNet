#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re

inputs   = [
    'out-eval/heuristic.csv',
    
    'out-eval/unet-lt-cosmic500-e50.csv',
    # 'out-eval/unet-l23-cosmic500-e50.csv',
    'out-eval/unet-l23-cosmic500-e50-t1.csv',
    # 'out-eval/unet-lt23-cosmic500-e50.csv',
    # 'out-eval/unet-explr-l23-cosmic500-e60.csv',
    # 'out-eval/unet-adam-l23-cosmic500-e50.csv',
    
    # 'out-eval/uresnet-explr-l23-cosmic500-e50.csv',
    # 'out-eval/uresnet-l23-cosmic500-e50-t1.csv',
    
    # 'out-eval/nestedunet-l23-cosmic500-e50.csv',
    ]
labels = [
    'Ref.',
    'DNN without MP',
    'DNN with MP',

    # "UNet",
    # "UResNet",
    # "UNet++",
    ]

markers = [
    'o',
    's',
    '*'
    # 'v',
    # '^',
    # '<',
    # '>',
]

metric_label = [
    "Pixel Efficiency",
    "Pixel Purity",
    "ROI Efficiency",
    "ROI Purity",
]

# 1: pix eff; 2: pix pur; 3: roi eff; 4: roi pur
imetric = 2

dfs = [pd.read_csv(i, sep=',', header=None) for i in inputs]
tags = [re.sub('-', ', ', i) for i in dfs[0][0].tolist()]

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
x = np.linspace(0,len(dfs[0][0].to_numpy())-1,len(dfs[0][0].to_numpy()))
for label, df, marker in zip(labels, dfs, markers):
    plt.plot(x,df[imetric].to_numpy(), '-'+marker, linewidth=3, markersize=14, label=label)

fontsize=26
plt.ylabel(metric_label[imetric-1], fontsize=fontsize)

ax.set_xlabel(r'$\theta_{xz}(V)$, $\theta_{xz}(U)$', fontsize=fontsize)
ax.legend(loc='best',fontsize=fontsize)
ax.grid()
plt.ylim([0, 1.5])
plt.yticks(fontsize=fontsize)
plt.xticks(x, tags, fontsize=fontsize)
plt.show()