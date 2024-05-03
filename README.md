# SAMSAM

## <!> This works is currently under development <!>

Experimenting with Segment Anything Model on digital pathology data

Comparison of different SAM based models for medical and digital pathology data on whole slide images.

This repository contains the code for web scraping data from cytomine platform, preprocessing the data, training models with different architectures and comparing the results.

Example of the results can be seen in notebooks

## TODO
- [ ] Implement training pipeline
- [ ] Implement train/test split
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

### Evaluation

As it is not mandatory to fine-tune the model, you can directly evaluate the model on your dataset. You can refer to evaluate_batch.py in this case

### Training

Not implemented yet

## <!> This works is currently under development <!>
