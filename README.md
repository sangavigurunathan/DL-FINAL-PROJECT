# DL-FINAL-PROJECT
Deep Learning for Skin Lesion Classification
This project applies Convolutional Neural Networks (CNNs) to classify skin lesions from dermoscopic images, targeting multiple types of infections. The model is trained to distinguish between bacterial, fungal, viral, and parasitic skin infections, aiming to aid dermatologists in accurate and automated diagnosis.

# Table of Contents
Introduction
Objective
Methodology
Data Description
Image Preprocessing
Model Selection
Model Training
Model Evaluation
Results
Future Work
References
Introduction
Skin conditions affect people globally, with early and accurate diagnosis being crucial for effective treatment. CNNs have shown potential in automating image analysis for medical diagnostics. This project leverages CNNs to analyze dermoscopic images, with the goal of providing a reliable, automated tool for skin disease diagnosis.

# Objective
The main goals of this project are:

To classify various skin lesions based on image features using CNNs.
To improve diagnostic accuracy and accessibility for dermatological assessments.
To evaluate model performance on a test dataset to ensure its reliability in real-world applications.

# Methodology
The classification process involves:

# Data Preprocessing: 
Standardizing image size and applying normalization and data augmentation.
# CNN Architecture: 
Using convolutional, pooling, and fully connected layers to capture key visual patterns for each disease.
# Training: 
Training the CNN model on a dermoscopic image dataset, optimizing performance with transfer learning and fine-tuning techniques.
# Evaluation: 
Using accuracy, precision, recall, and F1-score to evaluate model performance on test data.
# Data Description
A dataset of dermoscopic images representing 8 skin disease classes was used, split into 80% training and 20% testing. The dataset contains 1,159 labeled images across conditions such as bacterial infections (e.g., cellulitis, impetigo), fungal infections, parasitic infections, and viral infections.

# Image Preprocessing
Image preprocessing includes:

Resizing images to 224x224 pixels.
Normalization of pixel values to stabilize training.
Data Augmentation (rotation, flipping, scaling) to improve model generalization.
# Model Selection
The CNN model architecture was chosen for its effectiveness in capturing patterns in visual data. Transfer learning was used with pre-trained models (e.g., VGG16, ResNet) to fine-tune on our dataset.

# Model Training
The model was trained using the preprocessed dataset, with adjustments to improve convergence and performance. Key configurations included batch size, learning rate, and epochs.

# Model Evaluation
The CNN model’s effectiveness was measured using:

Accuracy: The percentage of correctly classified images.
Precision: The ratio of true positives to all positives predicted.
Recall: The proportion of true positives out of actual positives.
F1-Score: A balanced metric combining precision and recall.
Results
The model achieved:

Training Accuracy: 97%
Testing Accuracy: 75%
These results indicate that the model is effective for general skin lesion classification but may require additional tuning for rare skin conditions.

# Future Work
Future improvements include:

Refining the CNN model architecture to increase accuracy for less common diseases.
Implementing model interpretability to improve transparency in predictions.
Exploring deployment possibilities for use in telemedicine or mobile health applications.
References
Detailed references for the literature and research papers consulted can be found in the project report.

