# Classifying Alzheimer's Disease on ADNI Dataset Using ConvNeXt Small

**Author:** Baden Forster (s4744261)

---

## Project Overview

The aim of this project is to make use of MRI scan images of brains in the ADNI dataset to classify the presence of Alzheimer's disease. Making use of the powerful convolutional neural network (CNN) ConvNeXt, we aim to achieve a predictive accuracy of at least 80% in testing.

ConvNeXt (Liu et al., 2022) is a modern CNN designed to match the performance of Vision Transformers (ViTs) while keeping the efficiency and learning patterns of CNNs. Based on the ResNet-50 architecture, it was updated using design ideas from transformer models.

## Environment Setup

- Download the ADNI dataset and place it in the current directory. It should have the following structure:

```
.
└── ADNI/
    └── AD_NC/
        ├── test/
        │   ├── AD
        │   └── NC
        └── train/
            ├── AD
            └── NC
```

- Download Anaconda3 or Miniconda3
- Install NVIDIA CUDA toolkit
    - Download and install the latest stable version
    - Follow the installation guide for your OS.
- Create `conda` environment:
```bash
conda create -n env python=3.11.5
conda activate env
``` 

- Download dependencies
```bash
pip install torch torchvision numpy pandas matplotlib Pillow scikit-learn
```
- This will install the following dependencies:
    - PyTorch 2.9.0
    - torchvision 0.23.0
    - numpy 2.3.4
    - pandas 2.3.3
    - matplotlib 3.10.7
    - pillow 12.0.0
    - scikit-learn 1.7.2

## Usage

### To train the model on the ADNI train set: 
```bash
python train.py
```
This will save the trained model state dictionary `training_log.csv`, along with a log of training history, a plot of learning rate history `lr_history.png` and a plot of train and validation loss `loss_history.png` to the current directory. 

Using this trained model, inference can be run on the entire test set or a specific image.
### To run inference on the entire test set:
```bash
python predict.py
```
This will save a log of each training image evaluated, its predicted label and its true label `test_results.csv` to the current directory. It will display the confusion matrix and save it to the current directory as `confusion_matrix.png` It will also print the test set accuracy to the console.

### To predict a single image:
```bash
python predict.py --single path/to/image.jpg
```
This will display a figure of the provided image along with its predicted label. 

## TODO: Model Architecture / Selection
- componenets of model
- layers
- tiny vs small


## Dataset

The dataset provided by Alzheimer's Disease Neuroimaging Initiative (ADNI) [[1](https://adni.loni.usc.edu/data-samples/adni-data/)] contains 30,520 images of MRI scans of brains, which are labeled into two categories: Alzheimer's Disease (AD) and Normal Control (NC). The training set contains 10,400 AD images and 11,120 NC images. The test set contains 4,460 AD images and 4,540 NC images.

The dataset is fairly balanced between AD and NC, which helps prevent bias toward one class during training. The dataset has a roughly 29% split to test data, which is a reasonable amount with respect to common machine learning research and practise.  

## Data Preprocessing

The `dataset.py` script contains a definition for the `ADNI_Dataset` class, and data transformations used for the test and train datasets. The `ADNI_Dataset` is used in the training and prediction scripts to load the images from the disk and supply them to the model.

### Transforms

Each of the `test_transform` and `train_transform` contain standard manipulations to the images so that they can be effectively used by the model. These include:
- Conversion to greyscale
- Resizing to a standard size $(244 \times 244)$
- Converstion to a PyTorch Tensor
- Normalisation

In addition to this, the `train_transform` contains multiple random transformations that are applied to the image during training to help increase generalisation:
- Random crop to reduce size
- Random small rotation up to 5 degrees
- Random translation by a scale up to 2%
- Random scaling in the range 95% to 105%
- Random color intensity variation

These random transformations aim to increase generalisation by making the model less sensitive to irrelevant variations and more robust to unseen data. The `test_transform` does not contain these random transformations because its purpose is to evaluate the model's true performance on unseen data, not manipulated data.

### Train / Validation split

The dataset is already split into roughly 70% training and 30% testing, but I decided to further split the training set into 20% for validation and 80% for training. This split is a good compromise between being large enough to fairly represent the data distribution while still being small enough to leave most of the data for actual model training. The validation set will be used to tune the model during training. The final data split is the following:
- Test: 9,000 (29.5%)
- Train: 17,216 (56.4%)
- Validation: 4,304 (14.1%)

## TODO: Training

- hypers
- alg
    - scheduler
    - save model with best val loss
- loss
- optimiser

## TODO: Results

- accuracy / evaluation 

- what happened during training
- - observations
- - overfitting / underfitting ?
- results on test set
- - confusion matrix

## TODO: Conclusion / extensions

- use other info from the patient: ct scans, medical history, other health measurements (ie blood tests, scans, medical imaging)
- family history
- 
## References

Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022). A ConvNet for the 2020s. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, 5578–5588. https://doi.org/10.1109/CVPR52688.2022.00547

Alzheimer’s Disease Neuroimaging Initiative (2025). *ADNI Data*. Retrieved from https://adni.loni.usc.edu/data-samples/adni-data/