# DeepCrossing with DNN and CrossNetMix Layer

## Overview
This repository contains a modified version of DeepCrossing, a deep learning-based recommendation model. The modification includes replacing the residual network (ResNet) with a deep neural network (DNN) and adding a CrossNetMix layer.

## Contents
1. **Model Architecture**: 
    - The original DeepCrossing architecture has been altered to incorporate a DNN instead of a residual network. Additionally, a CrossNetMix layer has been introduced.
2. **Dependencies**: 
    - This project utilizes Python with PyTorch, depending on the preferred deep learning framework.
3. **Usage**:
    - Train the model using movielens1m.
    - Evaluate the model's performance using standard evaluation metrics.
4. **Example Code**:
    - Example scripts for training, evaluating, and using the modified DeepCrossing model are provided.

## Model Architecture
The modified DeepCrossing model architecture is as follows:
- **Input Layer**: Receives input features representing users and items.
- **DNN Layers**: A deep neural network replaces the original residual network. It captures complex feature interactions.
- **CrossNetMix Layer**: This additional layer enhances feature interactions and learning capabilities by incorporating cross networks with a mixture of cross products.
- **Output Layer**: Generate prediction in the out_dim dimension based on learning feature representation.

## Dependencies
- Python 3.10
- PyTorch
- NumPy
- Pandas
- torch4keras








