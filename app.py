# app.py - Live Deepfake Detector using YOUR trained SVM model
from flask import Flask, request, jsonify
from flask_cors import CORS
import io
import numpy as np
from pydub import AudioSegment
import librosa
import joblib
import os

# Force FFmpeg path to avoid warnings/errors
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\ffmpeg-8.0.1-essentials_build\ffmpeg-8.0.1-essentials_build\bin"

app = Flask(__name__)
CORS(app)

# Load your trained model and scaler
try:
    model = joblib.load("svm_model.pkl")      # Your current training output
    scaler = joblib.load("scaler.pkl")        # Your current training output
    print("✅ Model & scaler loaded successfully!")
except:
    print("❌ Model files missing!")
    exit(1)

# def extract_mfcc_features(y, sr, n_mfcc=13, n_fft=2048, hop_length=512):
#     """
#     EXACTLY matches your main.py training feature extraction
#     Returns 13 mean MFCC coefficients
#     """
#     try:
#         mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
#         return np.mean(mfccs.T, axis=0)  # Shape: (13,)
#     except Exception as e:
#         print(f"MFCC extraction error: {e}")
#         return None


def extract_advanced_features(audio_bytes):
    """🎯 SAME 48+ features for prediction"""
    try:
        audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
        
        mfcc = np.mean(librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13), axis=1)
        delta_mfcc = np.mean(librosa.feature.delta(mfcc), axis=1)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio_data, sr=sr))
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio_data))
        chroma = np.mean(librosa.feature.chroma(y=audio_data, sr=sr), axis=1)
        contrast = np.mean(librosa.feature.spectral_contrast(y=audio_data, sr=sr), axis=1)
        
        features = np.concatenate([mfcc, delta_mfcc, 
                                 [spectral_centroid, spectral_rolloff, zcr],
                                 chroma, contrast])
        return features
    except:
        return None


@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file received"}), 400

    audio_file = request.files['audio']
    audio_bytes = audio_file.read()

    print(f"📥 Received audio: {len(audio_bytes)/1024:.1f} KB")

    if len(audio_bytes) < 15000:
        return jsonify({"error": "Recording too short. Speak for 4+ seconds!"}), 400

    try:
        # Load browser audio (WebM/Opus) using pydub + FFmpeg
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        print(f"✅ Loaded: {audio.frame_rate}Hz, {audio.channels} channel(s), {len(audio)/1000:.1f}s")

        # Convert to mono numpy array
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        if audio.channels > 1:
            samples = samples.reshape((-1, audio.channels))
            samples = np.mean(samples, axis=1)

        # Normalize
        max_val = np.max(np.abs(samples))
        if max_val > 0:
            samples /= max_val

        sr = audio.frame_rate

        # Resample to 44100 Hz to match typical training files
        target_sr = 44100
        if sr != target_sr:
            samples = librosa.resample(samples, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
            print(f"   Resampled to {sr}Hz for consistent MFCC")

        # Extract exactly 13 MFCC features
        features = extract_advanced_features(audio_bytes)
        if features is None or len(features) != 13:
            return jsonify({"error": "Failed to extract MFCC features"}), 500

        # Predict using your trained model
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        confidence = round(max(probabilities) * 100, 1)  # 0–100%

        result = "REAL" if prediction == 0 else "FAKE"

        print(f"🤖 Prediction: {result} | Confidence: {confidence}%")

        return jsonify({
            "prediction": result,
            "confidence": confidence
        })

    except Exception as e:
        print(f"💥 Processing error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Audio processing failed. Try recording again."}), 500

if __name__ == '__main__':
    print("🚀 Deepfake Audio Detector (Using Your Trained SVM Model)")
    print("   Ensure svm_model.pkl and scaler.pkl are in this folder!")
    app.run(host='0.0.0.0', port=5000, debug=False)