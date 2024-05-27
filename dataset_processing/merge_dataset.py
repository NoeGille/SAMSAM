import os
import shutil
from typing import List

from tqdm import tqdm


def merge_datasets(root:str, datasets:List[str], verbose:bool=True) -> None:
    '''Merge n annotation_wise_scraper dataset into one. Copy them in a new directory.
    It is used to create large scale dataset for training and evaluation.
    root: str, path to the new directory where the datasets will be copied.
    datasets: List[str], list of paths to the datasets to merge.'''
    for i, dataset_path in tqdm(enumerate(datasets), total=len(datasets), desc='Merging datasets', disable=not verbose):
        dataset_files = get_file_path_list(dataset_path)
        for file_path in dataset_files:
            file_name = file_path.split('/')[-1]
            new_file_path = f'{root}/processed/{i}_{file_name}'
            copy_file(file_path, new_file_path)

def get_file_path_list(root:str) -> List[str]:
    '''Get the list of files in a directory.
    root: str, path to the directory
    Returns: List[str], list of file paths.'''
    return [f'{root}/{file}' for file in os.listdir(root)]

def copy_file(file_path:str, new_file_path:str) -> None:
    '''Copy a file to a new location.
    file_path: str, path to the file to copy
    new_file_path: str, path to the new location.'''
    os.makedirs(new_file_path, exist_ok=True)
    shutil.copy(file_path + '/img.jpg', new_file_path + '/img.jpg')
    shutil.copy(file_path + '/mask.jpg', new_file_path + '/mask.jpg')

if __name__ == '__main__':
    root = '../../datasets/merged/'
    datasets = ['../../datasets/camelyon16.1/processed/', '../../datasets/LBTD-NEO04/processed/']
    merge_datasets(root, datasets)