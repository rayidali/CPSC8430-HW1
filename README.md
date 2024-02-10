# CPSC8430-HW1
This repository encompasses Python implementations designed to elucidate core Deep Learning principles through practical experiments and simulations. The projects within offer a hands-on exploration of neural network architecture, function approximation, and the intricacies of training models on real-world data.

Project Overview
Simulating Functions with Neural Networks
This segment focuses on approximating a sinusoidal function, employing neural networks with varied architectural complexities to demonstrate:

Tensor operations and transformations
Utilizing various loss functions and calculating accuracy
Implementing activation functions
Understanding the impact of training epochs
MNIST Dataset Classification
We delve into the classic task of handwriting recognition using the MNIST dataset, constructing and training a neural network model. Key concepts illustrated include:

Convolutional Neural Networks (CNNs) for feature extraction
Dense (fully connected) layers for classification
Enhancing model performance with MaxPooling layers
Regularization techniques such as Dropout and weight decay to prevent overfitting
Dimensionality Reduction with PCA
Applying Principal Component Analysis (PCA) to the learned weights, this experiment aims to visualize the dimensionality reduction and identify pivotal weights in the model's learning process:

Extraction and analysis of model and layer-specific weights
Visualization of weight importance and distribution
Sensitivity Analysis in Neural Networks
By training models with varying batch sizes on the MNIST dataset, we investigate the relationship between model sensitivity and batch size, highlighting:

The effect of batch size on model training and sensitivity
Identification of an ideal batch size for optimal sensitivity and performance
Weight Interpolation and Generalization
Exploring the generalization capabilities of models through weight interpolation, this section provides insights into:

The process and impact of weight interpolation on model generalization
Systematic collection and analysis of model weights for interpolation
Installation Requirements
To replicate the experiments and run the code in this repository, ensure you have the following Python packages installed:

PyTorch: For building and training neural network models
Pandas: For data manipulation and analysis
NumPy: For numerical computations and operations
Scikit-learn: Specifically for applying PCA
Matplotlib: For plotting and visualizing data
Getting Started
Clone this repository to your local machine.
Ensure you have the required Python packages installed.
Navigate to the specific project directories to find the Jupyter notebooks or Python scripts.
Run the notebooks/scripts to observe the experiments and results firsthand.
