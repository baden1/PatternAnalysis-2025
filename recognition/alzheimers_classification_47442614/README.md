# Classifying Alzheimer's Disease on ADNI Dataset Using ConvNeXt Small

**Author:** Baden Forster (s4744261)

---

# Table of Contents

## Project Overview

The aim of this project is to make use of MRI scan images of brains in the ADNI dataset to classify the presence of Alzheimer's disease. Making use of the powerful convolutional neural network (CNN) ConvNeXt, we aim to achieve a predictive accuracy of at least 80% in testing.

ConvNeXt (Liu et al., 2022) is a modern CNN designed to match the performance of Vision Transformers (ViTs) while keeping the efficiency and learning patterns of CNNs. Based on the ResNet-50 architecture, it was updated using design ideas from transformer models.

## Model Architecture / Selection



## Dataset

The dataset provided by Alzheimer's Disease Neuroimaging Initiative (ADNI) [[1](https://adni.loni.usc.edu/data-samples/adni-data/)] contains many images of MRI scans of brains, which are labeled into two categories: Alzheimer's Disease (AD) and Normal Control (NC). The training set contains 10,400 AD images and 11,120 NC images. The test set contains 4,460 AD images and 4,540 NC images.

The dataset is fairly balanced between AD and NC, which helps prevent bias toward one class during training. The dataset has a roughly 29% split to test data, which is a reasonable amount with respect to common machine learning research and practise.  

## Data Preprocessing

## Training 

## Results

- what happened during training
- - observations
- - overfitting / underfitting ?
- results on test set
- - confusion matrix

## Usage

## Conclusion / extensions

## References

Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022). A ConvNet for the 2020s. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, 5578–5588. https://doi.org/10.1109/CVPR52688.2022.00547

Alzheimer’s Disease Neuroimaging Initiative (2025). *ADNI Data*. Retrieved from https://adni.loni.usc.edu/data-samples/adni-data/