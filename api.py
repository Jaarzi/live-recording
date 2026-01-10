from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile
import joblib
import librosa
import numpy as np


app = Flask(__name__)
CORS(app)


# Load One-Class model
# model = joblib.load("ultimate_deepfake_model.pkl")
# scaler = joblib.load("ultimate_scaler.pkl")

model = joblib.load("perfection_detector.pkl")
scaler = joblib.load("perfection_scaler.pkl")


def extract_features(audio_path):
    """Extract 78+ ADVANCED features matching ULTIMATE training"""
    try:
        audio, sr = librosa.load(audio_path, sr=16000, duration=4.0, mono=True)
        if len(audio) < 10000:
            return None
        
        features = []
        
        # MFCC (39 features)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        features.extend(np.mean(mfcc, axis=1))
        features.extend(np.mean(librosa.feature.delta(mfcc), axis=1))
        features.extend(np.mean(librosa.feature.delta(mfcc, order=2), axis=1))
        
        # Spectral (4 features)
        features.extend([np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))])
        features.extend([np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))])
        features.extend([np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))])
        features.extend([np.mean(librosa.feature.spectral_flatness(y=audios))])
        
        # Temporal (2 features)
        features.extend([np.mean(librosa.feature.zero_crossing_rate(audio))])
        features.extend([np.mean(librosa.feature.rms(y=audio))])
        
        # Chroma + Contrast (19 features)
        chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr), axis=1)
        contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr), axis=1)
        features.extend(chroma[:12])
        features.extend(contrast[:7])
        
        return np.array(features)
    except:
        return None


@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file'}), 400
   
    audio_file = request.files['audio']
    if not audio_file.filename.lower().endswith(('.wav', '.webm', '.m4a')):
        return jsonify({'error': 'Only audio files allowed'}), 400
   
    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name
   
    try:
        print(f"🎤 Processing: {audio_file.filename}")
        features = extract_features(tmp_path)
       
        if features is None:
            return jsonify({'error': 'Audio too short or invalid'}), 400
       
        # ONE-CLASS PREDICTION (IsolationForest)
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]  # 1=REAL, -1=FAKE
        anomaly_score = model.decision_function(features_scaled)[0]  # Higher = more REAL
       
        # Convert to standard format
        is_real = prediction == 1
        confidence = abs(anomaly_score)  # Distance from real pattern
       
        print(f"🤖 Result: {'REAL' if is_real else 'FAKE'} (confidence: {confidence:.3f})")
       
        return jsonify({
            'prediction': 'real' if is_real else 'fake',
            'confidence': float(confidence),
            'anomaly_score': float(anomaly_score),
            'is_real_pattern': is_real
        })
       
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500
   
    finally:
        os.unlink(tmp_path)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model': 'oneclass_deepfake'})


if __name__ == '__main__':
    print("🚀 One-Class Deepfake Detector API")
    print("✅ Real audio → Matches learned patterns")
    print("✅ Fake audio → Detected as anomaly")
    app.run(host='0.0.0.0', port=5000, debug=True)