# This script provides a way to evaluate the performance of SAM on a dataset.
# Please checkout https://github.com/facebookresearch/segment-anything to install segment anything
import matplotlib.pyplot as plt
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry
from sklearn.metrics import f1_score
from tqdm import tqdm

from dataset_processing.chu_anapath import ChuAnapath
from utils.config import load_config

config = load_config('config.toml')
model_type = config['sam']['model_type']
checkpoint_path = config['sam']['checkpoint_path']
dataset_path = config['cytomine']['dataset_path']
prompt_type = None if config['evaluation']['prompt_type'] == 'None' else config['evaluation']['prompt_type']
n_points = config['evaluation']['n_points']
device = config['sam']['device']


def evaluate(predictor:SamPredictor, dataset:ChuAnapath):
    '''Evaluate the model on the dataset'''
    dice_scores = []
    for i in tqdm(range(len(dataset))):
        img, mask, prompt = dataset[i]
        predictor.set_image(img)
        prediction, _, _ = predictor.predict(prompt, np.ones(prompt.shape[0]), multimask_output=False)
        dice_scores.append(f1_score(np.array(prediction.flatten(), dtype=mask.dtype), mask.flatten()))
    return dice_scores

if __name__ == '__main__':
    dataset = ChuAnapath(dataset_path, prompt_type=prompt_type, verbose=True, n_points=n_points)
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path).to(device)
    predictor = SamPredictor(sam)
    dice_score = evaluate(predictor, dataset)
    print(f'Mean Dice score: {np.mean(dice_score)}')