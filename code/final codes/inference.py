import io
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf

SR = 22050
SEG_SECONDS = 5.0
HOP_SECONDS = 2.5
N_MFCC = 20
N_FFT = 2048
HOP_LENGTH = 512
MAX_FRAMES = librosa.time_to_frames(SEG_SECONDS, sr=SR, hop_length=HOP_LENGTH)


def _segment_audio(y, seg_seconds, hop_seconds, sr):
    seg_len = int(seg_seconds * sr)
    hop_len = int(hop_seconds * sr)
    if len(y) < seg_len:
        return [np.pad(y, (0, seg_len - len(y)))]
    segments = []
    for start in range(0, len(y) - seg_len + 1, hop_len):
        segments.append(y[start:start + seg_len])
    return segments


def _mfcc_segment(y, sr):
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    if mfcc.shape[1] < MAX_FRAMES:
        pad = MAX_FRAMES - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad)), mode="constant")
    else:
        mfcc = mfcc[:, :MAX_FRAMES]
    return mfcc.astype(np.float32)


def _load_audio_from_bytes(payload):
    with sf.SoundFile(io.BytesIO(payload)) as f:
        y = f.read(dtype="float32")
        sr = f.samplerate
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return y


def model_fn(model_dir):
    model_path = Path(model_dir) / "gtzan_cnn.h5"
    model = tf.keras.models.load_model(model_path)
    mean = np.load(Path(model_dir) / "mfcc_mean.npy")[0]
    std = np.load(Path(model_dir) / "mfcc_std.npy")[0]
    classes = np.load(Path(model_dir) / "classes.npy", allow_pickle=True)
    return {"model": model, "mean": mean, "std": std, "classes": classes}


def input_fn(request_body, request_content_type):
    if request_content_type in ("audio/wav", "audio/x-wav", "application/octet-stream"):
        return _load_audio_from_bytes(request_body)
    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model_bundle):
    model = model_bundle["model"]
    mean = model_bundle["mean"]
    std = model_bundle["std"]
    classes = model_bundle["classes"]

    segments = _segment_audio(input_data, SEG_SECONDS, HOP_SECONDS, SR)
    feats = [_mfcc_segment(seg, SR) for seg in segments]
    X = np.stack(feats)
    X = (X - mean) / (std + 1e-8)
    X = X[..., np.newaxis]

    probs = model.predict(X, verbose=0)
    avg_probs = probs.mean(axis=0)
    pred_idx = int(avg_probs.argmax())

    return {
        "label": str(classes[pred_idx]),
        "score": float(avg_probs[pred_idx]),
        "probs": {str(classes[i]): float(avg_probs[i]) for i in range(len(classes))},
    }


def output_fn(prediction, response_content_type):
    if response_content_type == "application/json":
        return json.dumps(prediction), response_content_type
    raise ValueError(f"Unsupported response content type: {response_content_type}")
