import torch
import torchaudio
import sounddevice as sd
import numpy as np
import queue
import threading
# [Copy DeepfakeCNN + predict_audio from above]

# Live mic test (3-second chunks)
def live_predict():
    duration = 3  # seconds
    sample_rate = 16000
    
    def callback(indata, frames, time, status):
        q.put(indata.copy())
    
    q = queue.Queue()
    stream = sd.InputStream(samplerate=sample_rate, channels=1, callback=callback)
    
    print("🎤 Speak now... (Press Ctrl+C to stop)")
    with stream:
        while True:
            audio = q.get()
            audio = audio.flatten().numpy()
            status, conf = predict_audio_from_array(audio)  # Adapt predict_audio
            print(f"LIVE: {status} ({conf:.2f})")

if __name__ == "__main__":
    live_predict()
