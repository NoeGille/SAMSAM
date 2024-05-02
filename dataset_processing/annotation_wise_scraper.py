'''Download cropped images around annotations from a Cytomine project.
This script is made to be used on any Cytomine project to download images and masks around annotations.
Crops around annotations are resized into 1024x1024 images.'''

import logging
import os
import sys
from argparse import ArgumentParser
from math import floor
from tomllib import load

import matplotlib.pyplot as plt
import numpy as np
from cytomine import Cytomine
from cytomine.models import (
    Annotation,
    AnnotationCollection,
    ImageInstance,
    ImageInstanceCollection,
)
from shapely import wkt
from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm


def load_config(config_path:str):
    '''Loads a config file in toml format. Returns a dictionary with the config values.'''
    with open(config_path, "rb") as f:
        config = load(f)
    return config

def get_roi_around_annotation(img:ImageInstance, annotation:Annotation, config:dict) -> tuple[int, int, int, int]:
    '''Returns a valid region of interest (ROI) of a gigapixel image around an annotation.
    img: ImageInstance, the image to crop
    annotation: Annotation, the annotation to crop around
    config: dict, configuration dictionary
    random_shift: int, random shift to select the ROI
    random_state: int, random state for reproducibility of random shift.
    Returns: x, y, width, height of the ROI'''
    random_shift = config['cytomine']['random_shift']
    random_state = config['cytomine']['random_state']
    zoom_out_factor = config['cytomine']['zoom_out_factor']
    minimum_size = config['sam']['input_size']
    assert zoom_out_factor >= 1, 'Zoom out factor must be greater or equal to one'
    assert random_shift < img.width and random_shift < img.height, 'Random shift must be smaller than the image size'
    # Get location of the annotation
    geometry = wkt.loads(annotation.location).bounds
    x = floor(geometry[0])
    y = floor(img.height - geometry[3])
    # Apply random shift
    annotation_width = geometry[2] - geometry[0]
    annotation_height = geometry[3] - geometry[1]
    annotation_size = max(annotation_width, annotation_height)
    size = floor(min([max(annotation_size * config['cytomine']['zoom_out_factor'], minimum_size), img.width, img.height]))
    np.random.seed(random_state)
    if random_shift > 0:
        shift = np.random.randint(0, (np.abs(random_shift)) * 2, size=(2,)) - random_shift
    else:
        shift = np.array([0, 0])
    h = (size - annotation_width) / 2 
    v = (size - annotation_height) / 2
    x = floor(geometry[0] - h + shift[0])
    y = floor(img.height - geometry[3] - v + shift[1])
    # Correct the ROI if it is out of the image
    x = min(x, img.width - size)
    y = min(y, img.height - size)
    x = max(0, x)
    y = max(0, y)
    return x, y, size, size

def download_images(config:dict):
    '''Downloads all images from a Cytomine projects around annotations.
    Requires a config dictionary (see config.toml and load_config function).
    Must be executed unside a with Cytomine() statement.'''
    img_collections = ImageInstanceCollection().fetch_with_filter("project", config['cytomine']['project_id'])
    dataset_path = '../' + config['cytomine']['dataset_path'] + 'processed/'
    input_size = config['sam']['input_size']
    print('Croping and downloading images')
    for i,img in tqdm(enumerate(img_collections), total=len(img_collections)):
        annotations = AnnotationCollection()
        annotations.project = config['cytomine']['project_id']
        annotations.image = img.id
        annotations.user = config['cytomine']['annotation_user_id']
        annotations.showWKT = True
        annotations.showMeta = True
        annotations.showGIS = True
        annotations.fetch()
        for a, annotation in enumerate(annotations):
            x, y, width, height = get_roi_around_annotation(img, annotation, config)
            img.window(x, y, width, height, dest_pattern=dataset_path + f'{i}_{a}/img', users=[config['cytomine']['annotation_user_id']], annotations=[annotation.id], max_size=1024)
            img.window(x, y, width, height, mask=True, dest_pattern=dataset_path + f'{i}_{a}/mask', users=[config['cytomine']['annotation_user_id']], annotations=[annotation.id], max_size=1024)
            if len(os.listdir(dataset_path + f'{i}_{a}')) == 0:
                os.rmdir(dataset_path + f'{i}_{a}')
                print(f'Error downloading image {img.originalFilename} ({img.id}), annotation {a}')
                continue
            img_array = plt.imread(dataset_path + f'{i}_{a}/img.jpg')
            mask_array = plt.imread(dataset_path + f'{i}_{a}/mask.jpg')
            if len(mask_array.shape) == 3 or mask_array.sum() == 0:
                os.remove(dataset_path + f'{i}_{a}/mask.jpg')
                os.remove(dataset_path + f'{i}_{a}/img.jpg')
                os.rmdir(dataset_path + f'{i}_{a}')
            # SAM input size is 1024x1024
            else:
                if img_array.shape[0] != input_size or img_array.shape[1] != input_size:
                    resized_img = F.resize(F.to_pil_image(img_array), (input_size, input_size))
                    resized_img.save(dataset_path + f'{i}_{a}/img.jpg')
                    resized_mask = F.resize(F.to_pil_image(mask_array), (input_size, input_size), interpolation=InterpolationMode.NEAREST)
                    resized_mask.save(dataset_path + f'{i}_{a}/mask.jpg')

    

if __name__ == '__main__':
    parser = ArgumentParser(description='Download cropped images around annotations from a Cytomine project.')
    parser.add_argument('--config', required=False, type=str, help='Path to the configuration file. Default: ../config.toml', default='../config.toml')
    args = parser.parse_args()
    keys = load_config('../keys.toml')
    config = load_config(args.config)
    with Cytomine(keys['host'], keys['public_key'], keys['private_key'], verbose=logging.ERROR) as cytomine:
        download_images(config)
from pandas.io.parsers.readers import TextFileReader

TextFileReader
