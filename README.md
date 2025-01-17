# Face Mask Detection with DenseNet121

This repository contains a complete pipeline for detecting whether people are wearing face masks in images, using a transfer learning approach with **DenseNet121** (pre-trained on ImageNet). The project is implemented in Python, leveraging libraries such as TensorFlow/Keras, scikit-learn, and more.

<br />

## Table of Contents
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Training Procedure](#training-procedure)
5. [Results & Analysis](#results--analysis)


<br />

## Overview
The goal of this project is to build a **Deep Learning** model capable of classifying images into two categories:
- **With Mask**
- **Without Mask**

This approach uses:
- A **transfer learning** method, loading **DenseNet121** with pre-trained ImageNet weights.
- Additional custom layers (Dense, Dropout) to adapt the model to a binary classification task.
- Data augmentation (rotation, zoom, shift, etc.) to improve model robustness.

<br />

## Dataset
1. **Source**: The dataset is publicly available on Kaggle and includes images of individuals with and without face masks.
2. **Structure**: 
   - `with_mask/`
   - `without_mask/`
3. **Split**: 
   - **Train**: 80% (of the total images)
   - **Validation**: 20% (of train data)
   - **Test**: 20% (of the total images)

**Note**: Steps to download and preprocess the data are included in the notebook (or script).

<br />

## Model Architecture
A summary of the model pipeline:
1. **Base Model**: [DenseNet121](https://keras.io/api/applications/densenet/) (without the top classification layer).
2. **Additional Layers**:
   - Global Average Pooling
   - Dropout
   - Several Dense layers with ReLU activation
   - Final Dense layer with Softmax activation for 2 classes
3. **Optimizer**: [Adam](https://keras.io/api/optimizers/adam/)
4. **Loss**: `categorical_crossentropy`

<br />

## Training Procedure
1. **Data Augmentation** with `ImageDataGenerator`:
   - Random rotation, width shift, height shift, zoom, horizontal flip, etc.
2. **Hyperparameters**:
   - Epochs: up to 100 (early stopping used)
   - Initial Learning Rate: 0.001
   - Batch Size: 32
3. **Callbacks**:
   - `EarlyStopping` with patience=25
   - `ReduceLROnPlateau` to reduce LR on plateau
   - `ModelCheckpoint` to save the best model

<br />

## Results & Analysis
- **Accuracy**: Up to ~99% on the test set
- **Precision, Recall, F1-score**: ~0.99–1.00 across classes
- **AUC**: ~0.999 on the test data
- Confusion Matrix shows very few misclassifications.

> **Note**: Real-world performance may vary depending on image quality, lighting conditions, mask types, etc.

<br />
