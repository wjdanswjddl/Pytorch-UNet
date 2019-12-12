#
# load.py : utils on generators / lists of ids to transform from strings to
#           cropped images and masks

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt


def load(file, event, tags):
  data = h5py.File(file, 'r')
  frames = []
  for tag in tags:
    frames.append(np.array(data['/%d/%s'%(event, tag)]))
  img = np.stack(frames, axis = 2)
  img = np.transpose(img, axes=[1, 0, 2])
  return img

def rebin(a, shape):
  sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
  if len(a.shape) == 3:
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1],a.shape[2]
  return a.reshape(sh).mean(3).mean(1)

def plot_and_mask(img, mask):
  fig = plt.figure()
  a = fig.add_subplot(1, 3, 1)
  a.set_title('CH1')
  frame_ma = np.ma.array(np.transpose(img[:,:,0], axes=[1, 0]))
  plt.imshow(np.ma.masked_where(frame_ma<=0,frame_ma), cmap="jet", origin='lower')
  # plt.colorbar()
  plt.grid()

  a = fig.add_subplot(1, 3, 2)
  a.set_title('CH2')
  frame_ma = np.ma.array(np.transpose(img[:,:,1], axes=[1, 0]))
  plt.imshow(np.ma.masked_where(frame_ma<=0,frame_ma), cmap="jet", origin='lower')
  # plt.colorbar()
  plt.grid()

  a = fig.add_subplot(1, 3, 3)
  a.set_title('CH3')
  frame_ma = np.ma.array(np.transpose(img[:,:,2], axes=[1, 0]))
  plt.imshow(np.ma.masked_where(frame_ma<=0,frame_ma), cmap="jet", origin='lower')
  # plt.colorbar()
  plt.grid()

  fig = plt.figure()
  # a = fig.add_subplot(1, 2, 1)
  # a.set_title('Image')
  # plt.imshow(np.transpose(img, axes=[1, 0, 2]), origin='lower')

  # b = fig.add_subplot(1, 2, 2)
  # b.set_title('Mask')
  plt.imshow(np.transpose(mask, axes=[1, 0])
  , origin='lower'
  # , aspect='auto'
  )
  # print("Mask non-zero",np.count_nonzero(mask))
  # plt.colorbar()
  plt.grid()
  plt.show()

def get_hwc_img(file, event, tags, scale, crop0, crop1, norm):
  """From a list of tuples, returns the correct cropped img"""
  im = load(file, event, tags)
  im = rebin(im, [im.shape[0]//scale[0],im.shape[1]//scale[1]])/norm
  im = im[crop0[0]:crop0[1], crop1[0]:crop1[1], :]
  return im

def get_hwc_imgs(file, events, tags, scale, crop0, crop1, norm):
  """From a list of tuples, returns the correct cropped img"""
  for event in events:
    yield get_hwc_img(file, event, tags, scale, crop0, crop1, norm)

def get_chw_imgs(file, events, tags, scale, crop0, crop1, norm):
  """From a list of tuples, returns the correct cropped img"""
  for event in events:
    im = get_hwc_img(file, event, tags, scale, crop0, crop1, norm)
    im = np.transpose(im, axes=[2, 0, 1])
    yield im

def get_masks(file, events, tags, scale, crop0, crop1, threshold):
  """From a list of tuples, returns the correct cropped img"""
  for event in events:
    im = load(file, event, tags)
    im = im.reshape(im.shape[0],im.shape[1])
    im = rebin(im, [im.shape[0]//scale[0],im.shape[1]//scale[1]])
    im = im[crop0[0]:crop0[1], crop1[0]:crop1[1]]
    im[im<=threshold] = 0
    im[im>threshold] = 1
    yield im