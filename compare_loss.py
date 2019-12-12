#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd




tags   = ['simchan-tune-ttl-th0/loss','simchan-tune-ttl-th50/loss','simchan-tune-ttl-th100/loss', 'mp-roi-test-1-scale-th100/loss', 'loss']
labels = ['TH: 0','TH: 50','TH: 100', 'MP3', 'MP2+MP3']
dfs = [pd.read_csv(tag+'.csv', sep=' ', header=None) for tag in tags]

epoch = 5
nsample = 95

for i in range(len(tags)) :
  for isample in range(nsample*epoch):
    dfs[i][0][isample] = dfs[i][0][isample]+isample//nsample

for i in range(len(tags)) :
    # plt.plot(dfs[i][0][nsample*(epoch-1):nsample*(epoch)],dfs[i][1][nsample*(epoch-1):nsample*(epoch)],'o',label=labels[i])
    plt.plot(dfs[i][0],dfs[i][1],'o',label=labels[i])
plt.legend(loc='best',fontsize=15)
plt.grid()
# plt.ylim(0,800)
plt.xlabel("percentage", fontsize=15)
plt.ylabel("loss", fontsize=15)
plt.show()
