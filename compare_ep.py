#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

inputs   = [
    'out-eval/heuristic.csv',
    
    'out-eval/unet-l23-cosmic500-e50.csv',
    'out-eval/unet-l23-cosmic500-e50-t1.csv',
    'out-eval/unet-l23-cosmic500-e50-t2.csv',
    'out-eval/unet-explr-l23-cosmic500-e60.csv',
    'out-eval/unet-adam-l23-cosmic500-e50.csv',
    
    # 'out-eval/uresnet-explr-l23-cosmic500-e50.csv',
    # 'out-eval/uresnet-l23-cosmic500-e50-t1.csv',
    
    # 'out-eval/nestedunet-l23-cosmic500-e50.csv',
    ]
labels = [
    'Ref',
    
    'UNet-SGD-LR0.1-e50-t0',
    'UNet-SGD-LR0.1-e50-t1',
    'UNet-SGD-ExpLR-e60',
    'UNet-Adam-LR0.1-e50-t1',
    
    # 'UResNet-ExpLR-e50',
    # 'UResNet-LR0.1-e50-V1',
    
    # 'UNet++-LR0.1-e50',
    ]

metric_label = [
    "Pixel Efficiency",
    "Pixel Purity",
    "ROI Efficiency",
    "ROI Purity",
]

# 1: pix eff; 2: pix pur; 3: roi eff; 4: roi pur
imetric = 1

dfs = [pd.read_csv(i, sep=',', header=None) for i in inputs]

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
x = np.linspace(0,len(dfs[0][0].to_numpy())-1,len(dfs[0][0].to_numpy()))
for label, df in zip(labels, dfs):
    print(df)
    tags = df[0].to_numpy()
    plt.plot(x,df[imetric].to_numpy(),label=label)

fontsize=18
plt.ylabel(metric_label[imetric-1], fontsize=fontsize)

ax.set_xlabel(r'$\theta_{xz}(V)$ - $\theta_{xz}(U)$', fontsize=fontsize)
ax.legend(loc='best',fontsize=fontsize)
ax.grid()
plt.ylim([0, 1.5])
plt.xticks(x, tags, fontsize=fontsize)
plt.show()