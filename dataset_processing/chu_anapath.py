import os
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class AbstractSAMDataset(Dataset, ABC):
    '''Abstract class for SAM dataset
    root: str, path to the dataset directory
    prompt_type: [None, 'points', 'boxes'], type of automatic annotation to use
    n_points: int, number of points to use for automatic annotation if prompt is 'points'.
    verbose: bool, if True, print progress messages
    random_state: int, random state for reproducibility

    Three arrays must be defined in the child class:
    - self.images: list of paths to the images
    - self.masks: list of paths to the masks
    - self.prompts: list of prompts, each prompt is a numpy array of shape (n_points, 2) if prompt_type is 'points' or a tuple of 4 integers if prompt_type is 'boxes'

    Methods to compute the prompts(points and boxes) are already implemented but can be overriden if needed.
    '''

    @abstractmethod
    def _load_data(self):
        '''Load images and masks'''
        pass

    def _load_prompt(self):
        '''Compute and load prompts for the dataset'''
        if self.prompt_type == None:
            self.prompts = [None for _ in range(len(self.images))]
        elif self.prompt_type == 'points':
            self.prompts = np.array([self._get_points(self.masks[i]) for i in tqdm(range(len(self.images)), desc='Computing prompts...', total=len(self.images), disable=not self.verbose)])
        elif self.prompt_type == 'box':
            self.prompts = np.array([self._get_box(self.masks[i]) for i in tqdm(range(len(self.images)), desc='Computing prompts...', total=len(self.images), disable=not self.verbose)])
        else:
            raise ValueError('Invalid prompt type')

    def _get_points(self, mask_path:str):
        '''Get n_points points from the mask'''
        mask = plt.imread(mask_path)
        idx = np.arange(mask.shape[0]*mask.shape[1])
        flatten_mask = mask.flatten()
        points = np.random.choice(idx, self.n_points, p=flatten_mask/flatten_mask.sum())
        x, y = np.unravel_index(points, mask.shape)
        points = np.stack((y, x), axis=1)
        return points

    def _get_box(self, mask_path:str):
        '''Get a box from the mask'''
        x_min, y_min, x_max, y_max = mask.width, mask.height, 0, 0
        mask = plt.imread(mask_path)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j]== 1:
                    x_min = min(x_min, i)
                    y_min = min(y_min, j)
                    x_max = max(x_max, i)
                    y_max = max(y_max, j)
        return (x_min, y_min, x_max, y_max)


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = plt.imread(self.images[idx])
        mask = plt.imread(self.masks[idx])
        prompt = self.prompts[idx]
        if self.transform:
            img, mask = self.transform(img, mask)
        return img, np.where(mask > 0, 1, 0), prompt

class ChuAnapath(AbstractSAMDataset):
    '''Chu Anapath dataset for segmentation.'''

    def __init__(self, root:str, transform=None, prompt_type:str=None, n_points:int=1, verbose:bool=False, random_state:int=None):
        '''Initialize ChuAnapath dataset.
        root: str, path to the dataset directory
        prompt: [None, 'points', 'boxes'], type of automatic annotation to use
        n_points: int, number of points to use for automatic annotation if prompt is 'points'.'''
        self.root = root
        self.transform = transform
        self.images = []
        self.masks = []
        self.prompts = []
        self.prompt_type = prompt_type
        self.verbose = verbose
        self.n_points = n_points
        if random_state is not None:
            torch.manual_seed(random_state)
        if self.verbose:
            print('Loading images and masks paths...')
        self._load_data()
        self._load_prompt()
        if self.verbose:
            print('Done!')

    def _load_data(self):
        '''Load images and masks'''
        for f in os.listdir(self.root + 'processed/'):
            for g in os.listdir(self.root + 'processed/' + f):
                if g.endswith('.jpg'):
                    if 'mask' in g:
                        self.masks.append(self.root + 'processed/' + f + '/' + g)
                    else:
                        self.images.append(self.root + 'processed/' + f + '/' + g)
