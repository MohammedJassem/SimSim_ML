# SimSim_ML

## Overview

Welcome to our machine learning project repository! This repository is designed to be a flexible framework for training and deploying various machine learning models. Currently, it includes implementations for segmentation and named entity recognition (NER) models.

## Getting Started

### Clone the Repository

```bash
git clone https://github.com/your-username/your-repo.git
```
cd your-repo
### Install Dependencies

Install the required dependencies for the project using the following command:
```bash
pip install -r requirements.txt
```
### Data Preparation

Prepare your data for segmentation and NER. For NER, use the following example training script:
```bash
python3 train.py --file_path /downloads/ner_datasetreference.csv --epochs 5 --train_batch_size 8 --valid_batch_size 4 --learning_rate 0.0001 --max_grad_norm 5
```
### Segmentation Model

rain your segmentation model in the segmentation directory.
### FastAPI API

To run the FastAPI API for inference, execute the following command:
```bash
uvicorn main:app --reload
```

Visit http://127.0.0.1:8000/docs in your browser to explore the API documentation and test the models.
Contributing
