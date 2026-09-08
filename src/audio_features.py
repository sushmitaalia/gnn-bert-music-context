import librosa
import numpy as np

TARGET_SR = 22050
SEGMENT_SECONDS = 5


def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)
    return y, sr


def split_audio(y, sr, segment_seconds=SEGMENT_SECONDS):
    segment_length = int(sr * segment_seconds)
    segments = []

    for start in range(0, len(y), segment_length):
        end = start + segment_length
        segment = y[start:end]

        if len(segment) == segment_length:
            segments.append(segment)

    return segments


def extract_chroma(segment, sr):
    chroma = librosa.feature.chroma_stft(y=segment, sr=sr)
    return chroma.mean(axis=1)


def preprocess_track(file_path):
    y, sr = load_audio(file_path)
    segments = split_audio(y, sr)

    features = np.array([
        extract_chroma(segment, sr)
        for segment in segments
    ])

    norms = np.linalg.norm(features, axis=1, keepdims=True)
    normalized = features / (norms + 1e-8)

    return normalized
