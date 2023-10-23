#!/usr/bin/env python

import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt


if __name__ == '__main__':

  fn = sys.argv[1]
  key = sys.argv[2]
  

  data = h5py.File(fn, 'r')
  f = data.get(key)
  if f is None:
    print('f is None')
    exit()
  frame = np.array(f)
  print(frame.shape)
  frame_ma = np.ma.array(frame)

  plt.gca().set_title(key)
  # plt.imshow(np.ma.masked_where(frame_ma<=0,frame_ma), cmap="rainbow", interpolation="none"
  # plt.imshow(frame_ma>0, cmap="viridis", interpolation="none"
  plt.imshow(frame_ma, cmap="bwr", interpolation="none"
  # , extent = [0 , 2560, 0 , 6000]
  , origin='lower'
  # , aspect='auto'
  # , aspect=0.8/4.7
  , aspect=0.1
  )
  # plt.colorbar()
  # plt.xlim([0, 1600])
  # plt.xlim([0, 800]) # U
  # plt.xlim([800, 1600]) # V
  plt.xlim([476, 952]) # V
  # plt.clim([0,1])
  # plt.clim([2300,2400]) # orig U, V
  # plt.clim([885,915]) # orig W
  # plt.clim([-4000,4000])
  plt.clim([-1,1])

  plt.grid() 
  plt.show()


