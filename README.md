# Model Soups of ViT's for Brain Tumor Classification
This repository contains a Python codebase for Brain Tumor Classification from Magnetic Resonance Imaging scans using Model Soups of Vision Transformers. This project was conducted for a Master thesis titled "Comparison of individual Vision Transformers and Model Soups for Brain Tumor Classification on Magnetic Resonance Images", conducted by Ivo van Dongen at Tilburg University.

## Model Soups
This project is inspired by the great paper ['Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time'](https://arxiv.org/abs/2203.05482) by Wortsman et al. (2022).


## Dataset
The dataset used in this thesis is the "Brain Tumor MRI Dataset" (version 1) from [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset). This dataset contains 7023 Magnetic Resonance Imaging (MRI) scans from the brains of real human subjects and no other (personal) data. The dataset is publicly available under a 'CC:0 Public Domain' license. The dataset combines the following three public datasets: "Figshare", "SARTAJ", and "Br35H".

## Vision Transformers
'ViT-B32', 'ViT-B16', and 'ViT-L16' architectures from the vit-keras library are trained.
