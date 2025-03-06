# e20-co542-classical-music-classification
Neural network-based classical music classification using MusicNet and GTZAN datasets with MFCC feature extraction.

🎵 Classical Music Classification with Neural Networks

This project focuses on classifying classical and non-classical music using a neural network trained on the MusicNet and GTZAN datasets. We use MusicNet for classical music and GTZAN for non-classical genres, enabling the model to distinguish between them effectively.

📌 Project Overview

Utilizes deep learning techniques to classify classical vs. non-classical music.
Processes raw audio files and extracts relevant features using MFCC (Mel-Frequency Cepstral Coefficients).
Trains a neural network model to recognize musical patterns across genres.


📂 Dataset

We use two datasets for training and evaluation:

MusicNet – Contains labeled recordings of classical music, used for the classical category.
GTZAN – A well-known dataset with multiple genres, from which we use non-classical tracks.


🛠️ Technologies Used

Python
TensorFlow – Deep learning framework for training the neural network.
Librosa – Audio analysis and feature extraction (MFCC, spectrograms).
NumPy & Pandas – Data manipulation and numerical operations.
Matplotlib & Seaborn – Data visualization.
Scikit-learn (sklearn) – Model evaluation and preprocessing tools.
tqdm – Progress bar visualization for processing tasks.
os – File handling and directory management.


🚀 Features

MFCC-based feature extraction to capture audio characteristics.
Preprocessing pipeline for handling MusicNet and GTZAN datasets.
Neural network model trained to classify classical vs. non-classical music.
Training and evaluation scripts with accuracy tracking and performance metrics.
