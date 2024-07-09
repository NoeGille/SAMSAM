# SAMSAM

## <!> This works is currently under development <!>

Experimenting with Segment Anything Model on digital pathology data

<<<<<<< HEAD
This project uses TOML config file with are not supported by standard Python library until Python11.

<!> This works is currently under development <!>
=======
Comparison of different SAM based models for medical and digital pathology data on whole slide images.

This repository contains the code for web scraping data from cytomine platform, preprocessing the data, training models with different architectures and comparing the results.

Example of the results can be seen in notebooks

## TODO
- [x] Implement training pipeline
- [x] Implement train/test split
- [ ] Implement interactive prompting tool for the user

## Installation

### Clone the repository

```bash
git clone https://github.com/NoeGille/SAMSAM.git
```

### Requirements

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the requirements.

```bash
pip install -r requirements.txt
```

## Project configuration

For easier configuration of the project, we provide a `config.toml` file in the root of the project. This file contains all the necessary configuration for the project.
For most executables, you can specify your own toml file which will be used instead of the default one (It has to follow the same structure).
```bash
python samsam.py --config path/to/your/config.toml
```

Furthermore, if you want to use scrapers script on the Cytomine platform for your own datasets, you need to create a keys.toml file in the root of the project. This file should contain the following information:
```toml
title = "Cytomine API keys"
host = "[cytomine link]"
public_key = "[your public]"
private_key = "[your private key]"
```
Do not share this file with anyone.
Please refers to the Cytomine API Documentation for more information. https://doc.uliege.cytomine.org/

## Usage

It is recommended to use the SAMDataset class from `dataset_processing/dataset.py` to load the data and give it to the model. The class handle the processing of the data including automatic annotation of the data.

SAM is used throught a child class named TrainableSam. This class simplifies the use of SAM in a training pipeline without changing the inner functioning of SAM.

### Evaluation

As it is not mandatory to fine-tune the model, you can directly evaluate the model on your dataset. You can refer to evaluate.py in this case

### Training

Training is done through the TrainableSam class. You can refer to train.py in this case. You can either train the whole model (including the image encoder) or only the prompt encoder and decoder. In the latter case, use `save_img_embeddings.py` to save the image embeddings then `use_img_embeddings option` in `config.toml` to skip the image encoder part for faster training.

### Thanks
For the loss function implementation, thanks to Ma et al. .
```bibtex	
@article{LossOdyssey,
title = {Loss Odyssey in Medical Image Segmentation},
journal = {Medical Image Analysis},
volume = {71},
pages = {102035},
year = {2021},
author = {Jun Ma and Jianan Chen and Matthew Ng and Rui Huang and Yu Li and Chen Li and Xiaoping Yang and Anne L. Martel}
doi = {https://doi.org/10.1016/j.media.2021.102035},
url = {https://www.sciencedirect.com/science/article/pii/S1361841521000815}
}
```
## <!> This works is currently under development <!>
>>>>>>> develop
