import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import h5_utils as h5u  # Add this import

class HDF5Dataset(Dataset):
    def __init__(self, files_img, files_mask, img_tags, mask_tags, indices, rebin, x_range, y_range, z_scale, truth_th):
        self.files_img = files_img # list of input files
        self.files_mask = files_mask # list of target files
        assert len(files_img) == len(files_mask)
        self.nfiles = len(files_img)
        self.img_tags = img_tags
        self.mask_tags = mask_tags
        self.indices = indices
        self.rebin = rebin
        self.x_range = x_range
        self.y_range = y_range
        self.z_scale = z_scale
        self.truth_th = truth_th
        
    def __len__(self):
        return self.nfiles * len(self.indices)
    
    def __getitem__(self, index):
        fileno = index // len(self.indices)
        this_ID = index % len(self.indices) + 1 + np.min(self.indices)

        imgs = list(h5u.get_chw_imgs(self.files_img[fileno], [this_ID], self.img_tags, self.rebin, self.x_range, self.y_range, self.z_scale))[0]
        masks = list(h5u.get_masks(self.files_mask[fileno], [this_ID], self.mask_tags, self.rebin, self.x_range, self.y_range, self.truth_th))[0]
        
        return torch.tensor(imgs, dtype=torch.float32), torch.tensor(masks, dtype=torch.float32)
