import torch
import torchaudio
import librosa
import soundfile as sf
import numpy as np

# [Copy DeepfakeCNN class from app.py]

model = DeepfakeCNN()
model.load_state_dict(torch.load("deepfake_cnn.pth", map_location="cpu"))
model.eval()

mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=64)
mel_db = torchaudio.transforms.AmplitudeToDB()

def test_audio(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    if len(audio) > 48000:
        audio
