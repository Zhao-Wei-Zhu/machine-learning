# REPLY-Machine Learning Project
## Introduction
Efficient document management is crucial in today's fast-paced business environment. Whitehall Reply, a leader in technological innovations for Public Administration, seeks to enhance productivity by automating the classification of scanned documents using advanced machine learning techniques. This project aims to classify documents into predefined categories, thus minimizing manual effort and improving operational efficiency.

## Project Objective
The primary goal is to develop an image classification model that accurately categorizes scanned documents into five distinct classes: resumes (CV), advertisements (AD), emails (EMAIL), handwritten documents (DOC), and others (OTHER). The dataset consists of 2000 manually labeled images, and the objectives include:

#### Developing models using Convolutional Neural Networks (CNNs), Transformers, and Autoencoders.
#### Employing advanced machine learning techniques to achieve high accuracy and efficiency.
#### Evaluating each model's performance using appropriate metrics and cross-validation methods.

## Methodology
### 1. Convolutional Neural Network (CNN)
#### Environment Setup
* orchvision: Provides tools for handling image data transformations, part of the PyTorch ecosystem.
* Hugging Face's libraries: 'transformers' and 'evaluate' for state-of-the-art model training and evaluation.
* scikit-image: For image processing tasks.
* pandas: For data management in structured format.
#### Dataset Preparation
* Data Cleaning: Standardized image sizes to 64x64 pixels, ensuring consistency in quality and clarity.
* Normalization: Adjusted pixel values for stability and training speed.
#### Model Strategy
* Architecture: Consists of convolutional layers capturing spatial hierarchies, pooling layers, and fully connected layers for classification.
* Activation Function: LeakyReLU to prevent the "dying ReLU" problem.
#### Training and Evaluation
* Loss Function: Cross-entropy loss.
* Optimizer: Stochastic Gradient Descent (SGD).
* Validation: Stratified K-fold cross-validation to ensure a robust estimate of model performance.
* Metrics: Accuracy and F1-score, focusing on macro-average to address class imbalances.
##### The training spanned 10 epochs, emphasizing F1-score optimization to maximize learning from the limited dataset and mitigate overfitting. The model demonstrated consistent improvement in accuracy and F1-score, achieving a validation accuracy of 0.875256 and an F1-score of 0.789842 by the final epoch.

### 2. Autoencoders
#### Model Evaluation
The evaluation was a challenging aspect due to the reliance on reconstruction error, which is not directly interpretable. To address this, images were manually labeled post-reconstruction, and their accuracy was assessed using KMeans clustering, comparing the labels with predominant labels in each cluster.

#### Implementation Details
The autoencoder.ipynb file contains four sections, each reflecting progressive enhancements to the model:

#### Autoencoder Benchmark
* Resolution: Images resized to 21x28 pixels.
* Loss Function: Binary cross-entropy.
* Accuracy: Initial model accuracy stood at 49%, with notably low accuracy for handwritten documents and emails.
* Clustering: Included an 'other' group, emails and handwritten documents were clustered in the same cluster.
#### Autoencoder 2
* Resolution: Enhanced to 300x400 pixels.
* Loss Function: Switched to mean squared error.
* Improvement: Increased accuracy to 56%; however, emails and handwritten documents remained clustered together.
#### Autoencoder 3
* Enhancements: Introduced a function to remove black borders and increased the cluster number to 25.
* Clustering Strategy: Adopted a hierarchical approach to merge clusters based on the dominant category in each cluster, successfully segregating handwritten documents and emails with higher number of clusters.
* Accuracy: Improved to 65.4%.
#### Autoencoder Cross Validation
* Method: Implemented k-fold cross-validation for hyperparameter tuning, but for the RAM burden this process is handled manually by tuning single parameter per time, the first tuning focused on batch size, the second on epochs, the third on latent dimension. After obtaining best parameters, the best cluster number is also tested and merged to obtain 4 main clusters.
* Outcome:  the hyperparameter tuning was not effective in terms of accuracy improvement, all of the three tuning achieved 65% of accuracy.

### 3. Transformer
##### (Note：Please use google colab open the file ‘TransferLearning(Colab)’, Or download the file ‘TransferLearning(Local)’  and dataset(URL inside the code file) to run the code.)
#### Environment Setup
* TensorFlow and Keras: Primary frameworks for constructing and training neural network models, ideal for complex image data tasks.
* MobileNetV2: A pre-trained model known for its efficiency, particularly suitable for mobile devices, offering a good balance between speed and accuracy.
* Matplotlib and Seaborn: For visualizing training processes and understanding model performance through graphical representations.
* NumPy: Essential for numerical data manipulation, crucial in processing image data.
#### Dataset Preparation
The dataset, organized into specific classes, is processed as follows:
* Image Data Generator: Used for augmenting training data with transformations such as rotations, shifts, and flips to prevent overfitting. The target size for images is set to 224x224 pixels.
* Validation Split: A portion of the dataset is reserved for validation to monitor and evaluate the model's performance during training.
* Batch Processing: The dataset is processed in batches of 32 to optimize memory usage and accelerate the training process.
#### Model Strategy
* Base Model: MobileNetV2 with customized layers for classification.
* Optimizer: The model is compiled using an Adam optimizer with a categorical crossentropy loss function.
* Training Techniques: Implements early stopping and learning rate reduction to optimize training and prevent overfitting.
* Monitoring: Training adjustments are monitored through callbacks focusing on validation loss and accuracy. The total epochs set for training are 15.
* Performance Visualization: Accuracy and loss trends are plotted to identify potential overfitting or underfitting.
* Evaluation Metrics: A classification report and confusion matrix are generated, analyzing precision, recall, and F1-scores for each class.

## Conclusion
CNNs were selected as the best approach due to their effectiveness in image processing and classification. While Autoencoders showed promise in unsupervised feature learning and Transformers in sequential data processing, CNNs provided the most reliable and efficient results for the document classification tasks.

## Future Work
Future improvements could involve utilizing pre-trained models, applying data augmentation techniques, and optimizing hyperparameters to enhance model performance further. This project sets a foundation for continuous advancements in document management automation within Whitehall Reply.
