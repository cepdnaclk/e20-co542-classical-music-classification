from flask import Flask, request, jsonify
import os
import librosa
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load Pretrained Genre Classification Model
model = tf.keras.models.load_model("music_classification_model(2).h5")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

GENRES = ["Rock", "Hip-Hop", "Jazz", "Classical", "Pop", "Metal", "Blues", "Reggae"]

def extract_features(file_path):
    """Extract MFCC features from an audio file."""
    y, sr = librosa.load(file_path, duration=30)  # Load 30 sec of audio
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0).reshape(1, -1)  # Reshape for model input

@app.route("/upload", methods=["POST"])
def upload_audio():
    if "audio" not in request.files:
        return jsonify({"message": "No file part"}), 400

    file = request.files["audio"]
    if file.filename == "":
        return jsonify({"message": "No selected file"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Extract Features and Predict Genre
    features = extract_features(filepath)
    prediction = model.predict(features)
    predicted_genre = GENRES[np.argmax(prediction)]  # Get highest probability genre

    return jsonify({"message": "File uploaded successfully!", "genre": predicted_genre})

if __name__ == "__main__":
    app.run(debug=True)
