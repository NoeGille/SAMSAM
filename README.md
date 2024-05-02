# SAMSAM

## <!> This works is currently under development <!>

Experimenting with Segment Anything Model on digital pathology data

Comparison of different SAM based models for medical and digital pathology data on whole slide images.

This repository contains the code for web scraping data from cytomine platform, preprocessing the data, training models with different architectures and comparing the results.

Example of the results can be seen in notebooks

## Project configuration

For easier configuration of the project, we provide a `config.toml` file in the root of the project. This file contains all the necessary configuration for the project.
For most executables, you can specify your own toml file which will be used instead of the default one (It has to follow the same structure).
```bash
python samsam.py path/to/your/config.toml
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

## <!> This works is currently under development <!>
