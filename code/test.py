import os
import numpy as np
import librosa
import sys
import tensorflow as tf  # Assuming TensorFlow is used

# Load pre-trained model
loaded_model = tf.keras.models.load_model("music_classification_model.h5")

def extract_mfcc(audio_path, sr=22050, n_mfcc=20, max_length=130):
    y, sr = librosa.load(audio_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    if mfcc.shape[1] < max_length:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_length - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :max_length]

    return mfcc

# Path to the audio file
#C:\Users\sheha\OneDrive\Documents\GitHub\e20-co542-classical-music-classification\code\test.py
audio_path = "Make-a-Wish-Calm-Classical-Music-chosic.com_.mp3"  # Replace with actual file path

if not os.path.exists(audio_path):
    sys.exit("Exiting, file not found!")

# Extract MFCC
mfcc = extract_mfcc(audio_path)

# Normalize (ensure mean_train & std_train are precomputed or manually set)
mean_train, std_train = 0, 1  # Adjust based on training data if needed
mfcc = (mfcc - mean_train) / np.where(std_train == 0, 1, std_train)

# Reshape for model input
mfcc = np.expand_dims(mfcc, axis=(0, -1))  # Shape: (1, 20, 130, 1)

# Make prediction
prediction = loaded_model.predict(mfcc)
predicted_label = (prediction > 0.5).astype(int)

# Print prediction
if predicted_label[0] == 1:
    print("Predicted: Classical music")
else:
    print("Predicted: Non-classical music")
