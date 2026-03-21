# Breast Cancer Risk Prediction Models

Unified repository implementing state-of-the-art image-based breast cancer risk prediction models. This project aims to simplify training and evaluation of multiple models using a single, unified pipeline, supporting both single- and multi-timepoint mammogram analysis.

## Table of Contents
1. 📘 [Overview](#overview)  
2. ⚙️ [Models Implemented](#models-implemented)  
3. 📚 [Comparison of Models](#comparison-of-models)  
4. 🔍 [Dataset Format](#dataset-format)  
5. 💻 [Installation](#installation)
6. ▶️ [Usage](#usage)  
7. 📄 [Citation](#citation) 
8. 📝 [License](#license) 

## Overview
 

## Models Implemented


## Comparison of Models
 
## Dataset Format
 

##  Installation
For reproducing the results follow the instructions below:

## Usage

**Important**: for each script in the `scripts` folder, make sure you update the paths to load the correct datasets and export the results in your favorite directory.

### 1) Requirements
Requirements are in the requirements.txt file

### 2) Pre-processing of the datasets
The preprocessing step ensures that the datasets are properly prepared before training.

The `preprocessing` folder contains  the necessary scripts to preprocess images and split the datasets into training, validation and test.

To preprocess the EMBED dataset, use: `preprocessing/preprocess_img_embed.py`

To preprocess the CSAW-CC dataset, use: `preprocessing/preprocess_img_csaw_cc.py`

To split the datasets into training, validation, and test sets, use: `preprocessing/split_data.py`

Create a CSV file describing your dataset by running the notebooks in the `notebooks` folder

### 3) Training 
Run the following script `scripts/train.sh`

### 4) Evaluation
Run the following script `scripts/test.sh`

## Citation


## License