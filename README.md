# Classification-using-EfficientNet

This repository contains a deep learning project for **multiclass image classification** with additional **attribute-based features**. The model combines pre-trained image features from **EfficientNet-B2** with attribute data to improve classification accuracy.

## Dataset
- **Training Data**: 3,926 images (up-sampled to 7,000)
- **Test Data**: 4,000 images
- **Classes**: 200 different classes (attributes of images)

## Methodology

### Data Balancing
- **Up-sampling**: The dataset was balanced using **up-sampling** to address class imbalances, increasing the training dataset from 3,926 to 7,000 images (35 images per class).
- **Preprocessing**: Images were resized to a consistent size to handle varying image dimensions and aspect ratios, ensuring the model could handle differences in image shapes.

### Attributes Handling
- **Feature Engineering**: Attribute features  were passed through a fully connected layer to refine them to 128 dimensions before being concatenated with image features.

### Data Challenges
- **Varying Image Sizes**: Images were resized to a fixed dimension to ensure uniformity.
- **Class Imbalance**: Techniques like **up-sampling** were used to balance the dataset and reduce bias toward majority classes.

## Model Overview
1. **EfficientNet-B2**: Utilized as the base model for extracting image features. It was pre-trained on **ImageNet** to leverage learned features for transfer learning.
2. **Attribute Layer**: A custom fully connected layer processes the 94-dimensional attribute data and reduces its dimensions.
3. **Concatenation**: Image and attribute features were concatenated, followed by another fully connected layer for classification .
4. **Final Layer**: A fully connected layer outputs the class predictions, enabling multi-class classification.

### Final Performance
- The model achieved an **accuracy of 66%** on the test dataset.

## Libraries Used
- **torch**: PyTorch framework for model building and training.
- **timm**: Library for accessing pre-trained models like EfficientNet.
- **torchvision**: For image transformations and preprocessing.
- **Pandas & NumPy**: For data manipulation.
- **TensorFlow/Keras**: Used for initial image loading and resizing (via `load_img`).
- **matplotlib & seaborn**: For data visualization and analysis.
