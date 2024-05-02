from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import torch
from segment_anything.build_sam import build_sam_vit_b
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_processing.dataset import ChuAnapath
from dataset_processing.preprocess import collate_fn
from utils.config import load_config

parser = ArgumentParser(description='Evaluate a batch of images using a trained model.')
parser.add_argument('--config', required=False, type=str, help='Path to the configuration file. Default: ../config.toml', default='../config.toml')
args = parser.parse_args()
config = load_config(args.config)
n_points = config.dataset.n_points
dataset_path = config.cytomine.dataset_path
batch_size = config.training.batch_size
prompt_type = {'points':config.dataset.points, 'box':config.dataset.box, 'neg_points':config.dataset.negative_points}
n_points = config.dataset.n_points
inside_box = config.dataset.negative_points_inside_box
points_near_center = config.dataset.points_near_center


dataset = ChuAnapath(dataset_path, prompt_type=prompt_type, n_points=n_points, verbose=True, to_dict=True, neg_points_inside_box=inside_box, points_near_center=points_near_center)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
model = build_sam_vit_b(config.sam.checkpoint_path)
model.to('cuda')
model.eval()
dice_scores = []
iou_scores = []
precision_scores = []
recall_scores = []
with torch.no_grad():
    for data, mask in tqdm(dataloader):
        pred = model(data, multimask_output=True)
        for i in range(len(pred)):
            best_pred = pred[i]['masks'][0][pred[i]['iou_predictions'].argmax()].cpu().numpy()
            y_pred = np.array(best_pred.flatten(), dtype=mask[i].dtype)
            y_true = mask[i].flatten()
            dice_scores.append(f1_score(y_true, y_pred))
            iou_scores.append(jaccard_score(y_true, y_pred))
            precision_scores.append(precision_score(y_true, y_pred, zero_division=1))
            recall_scores.append(recall_score(y_true, y_pred, zero_division=0))
print(f'Mean Dice score: {np.mean(dice_scores)}')
print(f'Mean IoU score: {np.mean(iou_scores)}')
print(f'Mean Precision score: {np.mean(precision_scores)}')
print(f'Mean Recall score: {np.mean(recall_scores)}')



