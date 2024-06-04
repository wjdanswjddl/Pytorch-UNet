import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import h5_utils as h5u  # Add this import

class HDF5Dataset(Dataset):
    def __init__(self, file_img, file_mask, img_tags, mask_tags, indices, rebin, x_range, y_range, z_scale, truth_th):
        self.file_img = file_img
        self.file_mask = file_mask
        self.img_tags = img_tags
        self.mask_tags = mask_tags
        self.indices = indices
        self.rebin = rebin
        self.x_range = x_range
        self.y_range = y_range
        self.z_scale = z_scale
        self.truth_th = truth_th
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        index = self.indices[idx]
        
        imgs = list(h5u.get_chw_imgs(self.file_img, [index], self.img_tags, self.rebin, self.x_range, self.y_range, self.z_scale))[0]
        masks = list(h5u.get_masks(self.file_mask, [index], self.mask_tags, self.rebin, self.x_range, self.y_range, self.truth_th))[0]
        
        return torch.tensor(imgs, dtype=torch.float32), torch.tensor(masks, dtype=torch.float32)
