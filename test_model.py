# -*- coding: utf-8 -*-
print("🎤 LIVE DEEPFAKE TESTER - Backend Only!")
print("=" * 50)

import os
import librosa
import numpy as np
import joblib
from datetime import datetime

# Load your trained model
model = joblib.load("perfection_detector.pkl")
scaler = joblib.load("perfection_scaler.pkl")


def extract_features(audio_path):
    """Same features as training"""
    try:
        audio, sr = librosa.load(audio_path, sr=16000, duration=3.0, mono=True)
        if len(audio) < 8000:
            return None
        
        mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13), axis=1)
        mfcc_delta1 = np.mean(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)), axis=1)
        mfcc_delta2 = np.mean(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13), order=2), axis=1)
        
        return np.concatenate([mfcc, mfcc_delta1, mfcc_delta2])
    except:
        return None

def test_audio(file_path):
    """Test single audio file"""
    print(f"\n🔍 Testing: {os.path.basename(file_path)}")
    
    features = extract_features(file_path)
    if features is None:
        print("❌ Audio too short/invalid")
        return
    
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)[0]  # 1=REAL, -1=FAKE
    anomaly_score = model.decision_function(features_scaled)[0]
    
    is_real = prediction == 1
    confidence = abs(anomaly_score)
    
    print(f"🎯 RESULT: {'🟢 REAL HUMAN' if is_real else '🔴 AI DEEPFAKE'}")
    print(f"📊 Confidence: {confidence:.3f} | Anomaly: {anomaly_score:.3f}")
    print(f"⏰ {datetime.now().strftime('%H:%M:%S')}")
    print("-" * 40)

# 🚀 MAIN TEST MENU
print("\n🎵 SELECT TEST OPTION:")
print("1. Test your voice files (real_audio/)")
print("2. Test AI files (deepfake_audio/)") 
print("3. Test ANY audio file")
print("4. Continuous live testing")
print("0. Exit")

while True:
    choice = input("\nEnter choice (0-4): ").strip()
    
    if choice == "1":
        # Test all YOUR real files
        if os.path.exists("real_audio"):
            for f in os.listdir("real_audio"):
                if f.lower().endswith(('.wav', '.flac', '.mp3')):
                    test_audio(os.path.join("real_audio", f))
        else:
            print("❌ real_audio/ folder not found")
    
    elif choice == "2":
        # Test all YOUR fake files
        if os.path.exists("deepfake_audio"):
            for f in os.listdir("deepfake_audio"):
                if f.lower().endswith(('.wav', '.flac', '.mp3')):
                    test_audio(os.path.join("deepfake_audio", f))
        else:
            print("❌ deepfake_audio/ folder not found")
    
    elif choice == "3":
        # Test single file
        file_path = input("Enter full path to audio file: ").strip().strip('"')
        if os.path.exists(file_path):
            test_audio(file_path)
        else:
            print("❌ File not found!")
    
    elif choice == "4":
        # Continuous testing
        print("\n🔄 CONTINUOUS MODE - Ctrl+C to stop")
        while True:
            file_path = input("Drop audio file path: ").strip().strip('"')
            if os.path.exists(file_path):
                test_audio(file_path)
            else:
                print("❌ File not found")
    
    elif choice == "0":
        print("👋 Goodbye!")
        break
    
    else:
        print("❌ Invalid choice!")
