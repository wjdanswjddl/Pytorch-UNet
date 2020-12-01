#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df_loss = pd.read_csv('checkpoints/loss-batch.csv', sep=',', header=None)
loss = df_loss[1].to_numpy()
print('loss mean = {} median = {}'.format(np.mean(loss), np.median(loss)))

sample_means = []
for i in range(1000) :
    sample = np.random.choice(loss, 50, replace=False)
    sample_means.append(np.mean(sample))

fontsize = 24
plt.hist(sample_means, bins=100)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xlabel('50 sample mean loss', fontsize=fontsize)
plt.show()