# machine-learning
## Introduction
Efficient document management is crucial in today's fast-paced business environment. Whitehall Reply, a leader in technological innovations for Public Administration, seeks to enhance productivity by automating the classification of scanned documents using advanced machine learning techniques. This project aims to classify documents into predefined categories, thus minimizing manual effort and improving operational efficiency.

## Project Objective
The primary goal is to develop an image classification model that accurately categorizes scanned documents into five distinct classes: resumes (CV), advertisements (AD), emails (EMAIL), handwritten documents (DOC), and others (OTHER). The dataset consists of 2000 manually labeled images, and the objectives include:

Developing models using Convolutional Neural Networks (CNNs), Transformers, and Autoencoders.
Employing advanced machine learning techniques to achieve high accuracy and efficiency.
Evaluating each model's performance using appropriate metrics and cross-validation methods.

## Methodology
### 1. Convolutional Neural Network (CNN)
Environment Setup
torchvision: Provides tools for handling image data transformations, part of the PyTorch ecosystem.
Hugging Face's libraries: 'transformers' and 'evaluate' for state-of-the-art model training and evaluation.
scikit-image: For image processing tasks.
pandas: For data management in structured format.
Dataset Preparation
Data Cleaning: Standardized image sizes to 64x64 pixels, ensuring consistency in quality and clarity.
Normalization: Adjusted pixel values for stability and training speed.
Model Strategy
Architecture: Consists of convolutional layers capturing spatial hierarchies, pooling layers, and fully connected layers for classification.
Activation Function: LeakyReLU to prevent the "dying ReLU" problem.
Training and Evaluation
Loss Function: Cross-entropy loss.
Optimizer: Stochastic Gradient Descent (SGD).
Validation: Stratified K-fold cross-validation to ensure a robust estimate of model performance.
Metrics: Accuracy and F1-score, focusing on macro-average to address class imbalances.

### 2. Autoencoders
Implementation Details
Autoencoder models detect latent features and reconstruct images from these features, minimizing reconstruction error.
Images are clustered using KMeans, and model accuracy is assessed by comparing original labels with cluster labels.
Improvements include resizing images, increasing cluster numbers, and removing black borders.

### 3. Transformer
Environment Setup
Utilizes TensorFlow and MobileNetV2, effective for deep learning tasks.
Data augmentation and batch processing for training efficiency.
Model Strategy
Base Model: MobileNetV2 with customized layers for classification.
Training: Employed early stopping and learning rate adjustments to optimize performance.

## Conclusion
CNNs were selected as the best approach due to their effectiveness in image processing and classification. While Autoencoders showed promise in unsupervised feature learning and Transformers in sequential data processing, CNNs provided the most reliable and efficient results for the document classification tasks.

## Future Work
Future improvements could involve utilizing pre-trained models, applying data augmentation techniques, and optimizing hyperparameters to enhance model performance further. This project sets a foundation for continuous advancements in document management automation within Whitehall Reply.
