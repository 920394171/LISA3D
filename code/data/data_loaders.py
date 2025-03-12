import os
import torch
import numpy as np
from torch.utils.data import Dataset
import h5py
import itertools
from scipy import ndimage

class LISA3D_Dataset(Dataset):
    def __init__(self, task, split='train', repeat=None, unlabeled=False, num_cls=14, transform=None, is_val=False):
        self.task = task
        self.split = split
        self.repeat = repeat
        self.unlabeled = unlabeled
        self.num_cls = num_cls
        self.transform = transform
        self.is_val = is_val
        
        # Set data directory based on task
        if task == 'synapse':
            self.data_dir = '/data/Synapse'
        elif task == 'amos':
            self.data_dir = '/data/AMOS'
        else:
            raise ValueError(f'Unknown task: {task}')
        
        # Get file list
        self.filename_list = self._get_filename_list()
        
        # Repeat dataset if needed
        if self.repeat is not None:
            n_repeat = int(self.repeat / len(self.filename_list)) + 1
            self.filename_list = list(itertools.chain.from_iterable(
                itertools.repeat(self.filename_list, n_repeat)))
            self.filename_list = self.filename_list[:self.repeat]
    
    def _get_filename_list(self):
        """Get list of filenames for the dataset"""
        if self.task == 'synapse':
            data_dir = os.path.join(self.data_dir, self.split)
            return sorted([os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith('.h5')])
        elif self.task == 'amos':
            data_dir = os.path.join(self.data_dir, self.split)
            return sorted([os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith('.h5')])
    
    def __len__(self):
        return len(self.filename_list)
    
    def __getitem__(self, idx):
        # Load data
        h5_file = h5py.File(self.filename_list[idx], 'r')
        image = h5_file['image'][:]
        
        # Convert to float and normalize
        image = image.astype(np.float32)
        image = self._normalize_image(image)
        
        # For unlabeled data, we don't need the label
        if self.unlabeled:
            sample = {'image': image}
            if self.transform:
                sample = self.transform(sample)
            return sample
        
        # Load label for labeled data
        label = h5_file['label'][:]
        label = label.astype(np.int64)
        
        sample = {'image': image, 'label': label}
        
        # Apply transformations
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def _normalize_image(self, image):
        """Normalize image to [0, 1] range"""
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        return image

# Compatibility with SKCDF
class Synapse_AMOS(LISA3D_Dataset):
    def __init__(self, **kwargs):
        super(Synapse_AMOS, self).__init__(**kwargs)
