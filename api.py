from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile
import joblib
import librosa
import numpy as np

app = Flask(__name__)
CORS(app)

# Load your trained model (from main.py)
model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")

def extract_mfcc_features(audio_path, n_mfcc=13, n_fft=2048, hop_length=512):
    try:
        audio_data, sr = librosa.load(audio_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        return np.mean(mfccs.T, axis=0)
    except:
        return None

# def extract_mfcc_features(audio_path, n_mfcc=13, n_fft=2048, hop_length=512):
#     try:
#         # Let librosa detect native format (fixes PySoundFile issue)
#         audio_data, sr = librosa.load(audio_path, sr=None, mono=True)
        
#         print(f"📊 Audio loaded: {len(audio_data)} samples, {sr}Hz")
        
#         # Check if audio is too short (ReactMic sometimes creates empty)
#         if len(audio_data) < 1000:
#             print("❌ Audio too short (<1sec)")
#             return None
        
#         mfccs = librosa.feature.mfcc(
#             y=audio_data, 
#             sr=sr, 
#             n_mfcc=n_mfcc, 
#             n_fft=min(n_fft, len(audio_data)//4),  # Adaptive FFT
#             hop_length=min(hop_length, len(audio_data)//10)
#         )
        
#         features = np.mean(mfccs.T, axis=0)
#         print(f"✅ MFCC success: {features.shape}")
#         return features
        
#     except Exception as e:
#         print(f"❌ MFCC detailed error: {str(e)}")
#         import traceback
#         traceback.print_exc()
#         return None



@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file'})
    
    audio_file = request.files['audio']
    if not audio_file.filename.lower().endswith('.wav'):
        return jsonify({'error': 'Only WAV files allowed'})
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name
    
    try:
        features = extract_mfcc_features(tmp_path)
        if features is None:
            return jsonify({'error': 'Audio processing failed'})
        
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        confidence = max(probabilities)
        
        return jsonify({
            'prediction': 'real' if prediction == 0 else 'fake',
            'confidence': float(confidence),
            'probabilities': {
                'real': float(probabilities[0]),
                'fake': float(probabilities[1])
            }
        })
    finally:
        os.unlink(tmp_path)


# @app.route('/predict', methods=['POST'])
# def predict():
#     print("🎤 Received audio file")
    
#     if 'audio' not in request.files:
#         print("❌ No audio file")
#         return jsonify({'error': 'No audio file'}), 400
    
#     audio_file = request.files['audio']
#     print(f"📁 File: {audio_file.filename}, Size: {len(audio_file.read())} bytes")
#     audio_file.seek(0)  # Reset file pointer
    
#     if not audio_file.filename.lower().endswith('.wav'):
#         return jsonify({'error': 'Only WAV files allowed'}), 400
    
#     # Save uploaded file
#     with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
#         audio_file.save(tmp.name)
#         tmp_path = tmp.name
#         print(f"💾 Saved to: {tmp_path}")
    
#     try:
#         print("🔄 Processing audio...")
#         features = extract_mfcc_features(tmp_path)
        
#         if features is None:
#             print("❌ Audio processing failed")
#             return jsonify({'error': 'Audio processing failed'}), 400
        
#         print(f"✅ Features shape: {features.shape}")
#         features_scaled = scaler.transform([features])
#         prediction = model.predict(features_scaled)[0]
#         probabilities = model.predict_proba(features_scaled)[0]
#         confidence = max(probabilities)
        
#         print(f"🤖 PREDICTION: {prediction}, Confidence: {confidence:.2f}")
        
#         return jsonify({
#             'prediction': 'real' if prediction == 0 else 'fake',
#             'confidence': float(confidence),
#             'probabilities': {
#                 'real': float(probabilities[0]),
#                 'fake': float(probabilities[1])
#             }
#         })
#     except Exception as e:
#         print(f"💥 Unexpected error: {e}")
#         return jsonify({'error': f'Processing error: {str(e)}'}), 500
#     finally:
#         os.unlink(tmp_path)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
