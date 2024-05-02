# This script provides a way to evaluate the performance of SAM on a dataset.
# Please checkout https://github.com/facebookresearch/segment-anything to install segment anything
import matplotlib.pyplot as plt
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry
from sklearn.metrics import f1_score
from tqdm import tqdm

from dataset_processing.dataset import ChuAnapath
from utils.config import load_config

config = load_config('config.toml')
model_type = config['sam']['model_type']
checkpoint_path = config['sam']['checkpoint_path']
dataset_path = config['cytomine']['dataset_path']
prompt_type = None if config['dataset']['prompt_type'] == 'None' else config['dataset']['prompt_type']
n_points = config['dataset']['n_points']
device = config['sam']['device']


def evaluate(predictor:SamPredictor, dataset:ChuAnapath, prompt_type:str='points'):
    '''Evaluate the model on the dataset'''
    dice_scores = []
    for i in tqdm(range(len(dataset))):
        img, mask, prompt = dataset[i]
        predictor.set_image(img)
        if prompt_type == 'points':
            predictions, qualities, _ = predictor.predict(prompt, np.ones(prompt.shape[0]))
        elif prompt_type == 'box':
            predictions, qualities, _ = predictor.predict(box=prompt)
        else:
            predictions, qualities, _ = predictor.predict()
        prediction = predictions[np.argmax(qualities)]
        dice_scores.append(f1_score(np.array(prediction.flatten(), dtype=mask.dtype), mask.flatten()))
    return dice_scores

if __name__ == '__main__':
    dataset = ChuAnapath(dataset_path, prompt_type=prompt_type, verbose=True, n_points=n_points, to_dict=False)
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path).to(device)
    predictor = SamPredictor(sam)
    dice_score = evaluate(predictor, dataset, prompt_type)
    print(f'Mean Dice score: {np.mean(dice_score)}')