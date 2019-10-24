#
# load.py : utils on generators / lists of ids to transform from strings to
#           cropped images and masks

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt


def load_h5(file, event, tag0, tag1):
  data = h5py.File(file, 'r')
  frame0 = np.array(data['/%d/%s'%(event, tag0)])
  frame1 = np.array(data['/%d/%s'%(event, tag1)])
  frame = np.stack((frame0, frame0, frame1), axis = 2)
  frame = np.transpose(frame, axes=[1, 0, 2])
  return frame


def plot_h5_and_mask(img, mask):
    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    a.set_title('Input image')
    plt.imshow(np.transpose(img, axes=[1, 0, 2]), origin='lower')

    b = fig.add_subplot(1, 2, 2)
    b.set_title('Output mask')
    plt.imshow(np.transpose(mask, axes=[1, 0]), origin='lower')
    plt.show()