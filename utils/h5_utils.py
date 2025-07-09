#
# load.py : utils on generators / lists of ids to transform from strings to
#           cropped images and masks

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

def kernel(f, sigma=4.5):
    sigma0 = 4.
    sigmat = np.sqrt(sigma**2 - sigma0**2)
    fsize = f.shape[0]
    ticks = np.linspace(0, fsize-1, fsize)
    ticks[ticks > fsize//2] = ticks[ticks > fsize//2] - fsize
    kernel = np.exp(-(ticks)**2 / (2*sigmat**2))
    return kernel / kernel.sum()

def convolve(f1, f2):
    '''
    Return the simple convolution of the two arrays using FFT+mult+invFFT method.
    '''
    # fftconvolve adds an unwanted time shift
    #from scipy.signal import fftconvolve
    #return fftconvolve(field, elect, "same")
    s1 = np.fft.fft(f1)
    s2 = np.fft.fft(f2)
    sig = np.fft.ifft(s1*s2)

    return np.real(sig)

def smear(f, smearing=4.5):
    f = f.T
    k = kernel(f, smearing)
    
    out = np.zeros(f.shape)
    for i in range(f.shape[1]):
        out[:, i] = convolve(f[:, i], k)
    
    out = out.T
    # case to float
    out = out.astype(np.float32)

    return out

def erase_wires(f, center=500, n=50):
    if n == 0:
        return f
    f[:, center-n//2: center+n//2] = 0
    return f


def load(file, event, tags, erase_center=500, erase_n=50):
  data = h5py.File(file, 'r')
  frames = []
  for tag in tags:
    f = data.get('/%d/%s'%(event, tag))
    if f is None:
      return None
    arr_f = np.array(f)
    arr_f = erase_wires(arr_f, center=erase_center, n=erase_n)
    frames.append(arr_f)
  img = np.stack(frames, axis = 2)
  img = np.transpose(img, axes=[1, 0, 2])
  return img

def rebin(a, shape):
  sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
  if len(a.shape) == 3:
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1],a.shape[2]
  return a.reshape(sh).mean(3).mean(1)

def plot_img(img):
  for ich in range(img.shape[2]) :
    fig = plt.figure()
    a = fig.add_subplot(1, 1, 1)
    a.set_title('CH{}'.format(ich))
    frame_ma = np.ma.array(np.transpose(img[:,:,ich], axes=[1, 0]))
    # plt.imshow(np.ma.masked_where(frame_ma<=0,frame_ma), cmap="bwr_r", origin='lower')
    plt.imshow(frame_ma, cmap="bwr", origin='lower')
    plt.clim(-1,1)
    # plt.colorbar()
    plt.grid()
  plt.show()

def plot_mask(mask, savename=""):
    plt.figure(figsize=(6,4))
    # plt.gca().set_title('Mask')
#    mask = mask[800:1200, :3500]
    plt.imshow(np.transpose(mask, axes=[1, 0])
    , cmap="bwr"
    , origin='lower'
    , aspect='auto'
    )
    # print("Mask non-zero",np.count_nonzero(mask))
    # plt.colorbar()
    plt.clim(-1,1)
    plt.xlabel("wire")
    plt.ylabel("tick")
    #  plt.xlim(950, 1350)
    #  plt.ylim(600,1800)
    #xlims=[950,1350], ylims=[600,1800]
    #plt.grid()
    #  plt.show()
    plt.savefig(savename+"predict_vis.png", bbox_inches="tight", dpi=200)
    print("figure saved")

def get_hwc_img(file, event, tags, scale, crop0, crop1, norm, erase_center=500, erase_n=50):
  """From a list of tuples, returns the correct cropped img"""
  im = load(file, event, tags, erase_center, erase_n)
  if im is None:
    return None
  #print("chw img shape", im.shape)
  im = rebin(im, [im.shape[0]//scale[0],im.shape[1]//scale[1]])/norm
  im = im[crop0[0]:crop0[1], crop1[0]:crop1[1], :]
  #print("chw rebin img shape", im.shape)
  return im

def get_hwc_imgs(file, events, tags, scale, crop0, crop1, norm, erase_center=500, erase_n=50):
  """From a list of tuples, returns the correct cropped img"""
  for event in events:
    im = get_hwc_img(file, event, tags, scale, crop0, crop1, norm, erase_center, erase_n)
    if im is None:
      continue
    yield im

def get_chw_imgs(file, events, tags, scale, crop0, crop1, norm, erase_center=500, erase_n=50):
  """From a list of tuples, returns the correct cropped img"""
  for event in events:
    im = get_hwc_img(file, event, tags, scale, crop0, crop1, norm, erase_center, erase_n)
    if im is None:
      continue
    im = np.transpose(im, axes=[2, 0, 1])
    yield im

def get_masks(file, events, tags, scale, crop0, crop1, threshold, smearsig=4.5, erase_center=500, erase_n=50):
  """From a list of tuples, returns the correct cropped img"""
  for event in events:
    im = load(file, event, tags, erase_center, erase_n)
    if im is None:
      continue

    im = smear(im.T[0].T, smearing=smearsig)

    im = im.reshape(im.shape[0],im.shape[1])
    im = rebin(im, [im.shape[0]//scale[0],im.shape[1]//scale[1]])
    im = im[crop0[0]:crop0[1], crop1[0]:crop1[1]]
    im[im<=threshold] = 0
    im[im>threshold] = 1
    yield im
