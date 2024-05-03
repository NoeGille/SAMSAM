'''Dataset class for SAM dataset to use in PyTorch. Please make sure that your images are 1024x1024 pixels to prevent any problems with the model performances.'''
import os
from abc import ABC, abstractmethod
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from cv2 import dilate, erode
from scipy.ndimage import distance_transform_edt
from torch.utils.data import Dataset
from tqdm import tqdm

from .preprocess import to_dict


class AbstractSAMDataset(Dataset, ABC):
    '''Abstract class for SAM dataset
    root: str, path to the dataset directory
    prompt_type: [None, 'points', 'box'], type of automatic annotation to use
    n_points: int, number of points to use for automatic annotation if prompt is 'points'.
    verbose: bool, if True, print progress messages
    random_state: int, random state for reproducibility

    Three arrays must be defined in the child class:
    - self.images: list of paths to the images
    - self.masks: list of paths to the masks
    - self.prompts: list of prompts, each prompt is a numpy array of shape (n_points, 2) if prompt_type is 'points' or a tuple of 4 integers if prompt_type is 'boxes'
    Moreover you need to specified some arguments in the child class:
    - self.transform: a callable to apply to the images and masks
    - self.prompt_type: Dict[str, bool], type of automatic annotation to use
    - self.to_dict: bool, if True, the __getitem__ method will return a dictionary with the image, the prompt and the mask. If False, it will return the image, the mask and the prompt.
    to_dict is especially useful when using Sam class directly instead of SamPredictor class.
    Methods to compute the prompts(points and boxes) are already implemented but can be overriden if needed.
    '''

    @abstractmethod
    def _load_data(self):
        '''Load images and masks'''
        pass

    def _load_prompt(self, zoom_out:float=1.0, n_points:int=1, inside_box:bool=False, near_center:float=-1, random_box_shift:int=0) -> dict[str, np.ndarray]:
        '''Compute and load prompts for the dataset'''
        prompts = {'points':[None for _ in range(len(self.images))], 
                   'box':[None for _ in range(len(self.images))], 
                   'neg_points':[None for _ in range(len(self.images))]}
        if self.prompt_type['neg_points']:
            prompts['neg_points'] = np.array([self._get_negative_points(self.masks[i], n_points, inside_box) for i in tqdm(range(len(self.images)), desc='Computing negative points...', total=len(self.images), disable=not self.verbose)])
        if self.prompt_type['points']:
            prompts['points'] = np.array([self._get_points(self.masks[i], n_points, near_center) for i in tqdm(range(len(self.images)), desc='Computing points...', total=len(self.images), disable=not self.verbose)])
        if self.prompt_type['box']:
            prompts['box'] = np.array([self._get_box(self.masks[i], zoom_out, random_box_shift) for i in tqdm(range(len(self.images)), desc='Computing boxes...', total=len(self.images), disable=not self.verbose)])
        self.prompts = prompts

    def _get_points(self, mask_path:str, n_points:int=1, near_center:float=-1):
        '''Get n_points points from the mask'''
        mask = plt.imread(mask_path)
        idx = np.arange(mask.shape[0]*mask.shape[1])
        flatten_mask = mask.flatten()
        if near_center <= 0:
            points = np.random.choice(idx, n_points, p=flatten_mask/flatten_mask.sum())
        else:
            distance = (distance_transform_edt(mask)**near_center).flatten()
            points = np.random.choice(idx, n_points, p=(distance/distance.sum()))
        x, y = np.unravel_index(points, mask.shape)
        points = np.stack((y, x), axis=1)
        return points

    def _get_negative_points(self, mask_path:str, n_points:int, inside_box:bool=False):
        '''Get n_points points outside the mask'''
        mask = plt.imread(mask_path)
        idx = np.arange(mask.shape[0]*mask.shape[1])
        flatten_mask = mask.flatten()
        if inside_box:
            dilation = 100 # dilation is applied to both box_mask and mask to avoid points near the true mask
            x_min, y_min, x_max, y_max = self._get_box(mask_path)
            box_mask = np.ones_like(mask)
            box_mask[y_min:y_max, x_min:x_max] = 0
            flatten_mask = erode(box_mask, np.ones((dilation, dilation))).flatten() + dilate(mask,np.ones((dilation, dilation))).flatten()
        points = np.random.choice(idx, n_points, p=(1 - flatten_mask)/((1 - flatten_mask).sum()))
        x, y = np.unravel_index(points, mask.shape)
        points = np.stack((y, x), axis=1)
        return points
    
    def _get_box(self, mask_path:str, zoom_out:float=1.0, random_box_shift:int=0) -> tuple[int, int, int, int]:
        '''Get a box from the mask
        zoom_out: float, factor to zoom out the box. Add a margin around the mask.'''
        assert zoom_out > 0, 'Zoom out factor must be greater than 0'
        mask = plt.imread(mask_path)
        x_min, y_min, x_max, y_max = mask.shape[0], mask.shape[1], 0, 0
        h_sum = mask.sum(axis=1)
        v_sum = mask.sum(axis=0)
        x_min = np.argmax(v_sum > 0)
        x_max = len(v_sum) - np.argmax(v_sum[::-1] > 0)
        y_min = np.argmax(h_sum > 0)
        y_max = len(h_sum) - np.argmax(h_sum[::-1] > 0)
        box_width = x_max - x_min
        box_height = y_max - y_min
        h_padding = (box_width * zoom_out - box_width)  / 2
        v_padding = (box_height * zoom_out - box_height) / 2
        shifts = [0, 0, 0, 0]
        if random_box_shift > 0:
            shifts = np.random.randint(-random_box_shift, 2*random_box_shift, 4)
        x_min = max(0, x_min - h_padding + shifts[0])
        x_max = min(mask.shape[0], x_max + h_padding + shifts[1])
        y_min = max(0, y_min - v_padding + shifts[2])
        y_max = min(mask.shape[1], y_max + v_padding + shifts[3])
        return int(x_min), int(y_min), int(x_max), int(y_max)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx:int) -> tuple[np.ndarray, np.ndarray, Sequence] | tuple[dict, np.ndarray]:
        img = plt.imread(self.images[idx])
        mask = plt.imread(self.masks[idx])
        prompt = {'points':self.prompts['points'][idx], 'box':self.prompts['box'][idx], 'neg_points':self.prompts['neg_points'][idx]}
        if self.transform:
            img, mask = self.transform(img, mask)
        if self.to_dict:
            return to_dict(img, prompt), np.where(mask > 0, 1, 0)
        return img, np.where(mask > 0, 1, 0), prompt

class SAMDataset(AbstractSAMDataset):
    '''Prepare a dataset for segmentation by Segment Anything Model. Checkout AbstractSAMDataset for more information'''

    def __init__(self, root:str, transform=None, prompt_type:dict={'points':False, 'box': False, 'neg_points':False}, n_points:int=1, zoom_out:float=1.0, verbose:bool=False, random_state:int=None, to_dict:bool=True, neg_points_inside_box:bool=False, points_near_center:float=-1, random_box_shift:int=0):
        '''Initialize SAMDataset class.
        root: str, path to the dataset directory
        prompt_type: Dict[str, bool], type of automatic annotation to use
        n_points: int, number of points to use for automatic annotation if prompt is 'points'.
        zoom_out: float, factor to zoom out the bounding box. Add a margin around the mask.'''
        self.root = root
        self.transform = transform
        self.images = []
        self.masks = []
        self.prompts = []
        self.prompt_type = prompt_type
        self.verbose = verbose
        self.n_points = n_points
        self.near_center = points_near_center
        self.inside_box = neg_points_inside_box
        self.to_dict = to_dict
        self.zoom_out = zoom_out
        if random_state is not None:
            torch.manual_seed(random_state)
        if self.verbose:
            print('Loading images and masks paths...')
        self._load_data()
        self._load_prompt(zoom_out=zoom_out, n_points=n_points, inside_box=neg_points_inside_box, near_center=points_near_center, random_box_shift=random_box_shift)
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
