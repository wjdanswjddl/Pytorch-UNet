#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np




# tags   = ['simchan-tune-ttl-th0/loss','simchan-tune-ttl-th50/loss','simchan-tune-ttl-th100/loss', 'mp-roi-test-1-scale-th100/loss', 'loss']
# labels = ['TH: 0','TH: 50','TH: 100', 'MP3', 'MP2+MP3']
tags   = ['l23-cosmic500-e50/loss']
labels = ['l23: 500 x 50']
dfs = [pd.read_csv(tag+'.csv', sep=' ', header=None) for tag in tags]

epoch = 50
nsample = 450

# for i in range(len(tags)) :
#   for isample in range(nsample*epoch):
#     dfs[i][0][isample] = dfs[i][0][isample]+isample//nsample

# for i in range(len(tags)) :
#     # plt.plot(dfs[i][0][nsample*(epoch-1):nsample*(epoch)],dfs[i][1][nsample*(epoch-1):nsample*(epoch)],'o',label=labels[i])
#     plt.plot(dfs[i][0],dfs[i][1],'o',label=labels[i])
# plt.legend(loc='best',fontsize=15)
# plt.grid()
# # plt.ylim(0,800)
# plt.xlabel("percentage", fontsize=15)
# plt.ylabel("loss", fontsize=15)
# plt.show()


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