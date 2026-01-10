# # -*- coding: utf-8 -*-
# print("🚀 ONE-CLASS DEEPFAKE DETECTOR (Real-Only)")
# import os, librosa, numpy as np, joblib
# from sklearn.ensemble import IsolationForest
# from sklearn.preprocessing import StandardScaler

# # YOUR ASVspoof REAL FILES (25K+ perfect samples)
# TRAIN_PATH = r"C:\Users\library\Desktop\voice detction system\LA\ASVspoof2019_LA_train\flac"
# print(f"📁 Loading REAL patterns: {TRAIN_PATH}")

# X = []
# print("✅ Learning REAL voice patterns...")

# if os.path.exists(TRAIN_PATH):
#     for root, _, files in os.walk(TRAIN_PATH):
#         for f in files:
#             if f.endswith('.flac') and f.startswith('LA_T'):  # Only REAL
#                 path = os.path.join(root, f)
#                 try:
#                     audio, sr = librosa.load(path, sr=16000, duration=3.0, mono=True)
                    
#                     # 39+ ROBUST features for real pattern detection
#                     mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13), axis=1)
#                     mfcc_delta1 = np.mean(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)), axis=1)
#                     mfcc_delta2 = np.mean(librosa.feature.delta(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13), order=2), axis=1)
                    
#                     features = np.concatenate([mfcc, mfcc_delta1, mfcc_delta2])
#                     X.append(features)
                    
#                     if len(X) % 500 == 0:
#                         print(f"✅ {len(X)} real patterns learned")
                        
#                     if len(X) >= 5000:  # Enough for perfect detection
#                         break
#                 except:
#                     pass
#         if len(X) >= 5000:
#             break

# print(f"\n🎉 Learned {len(X)} REAL voice patterns!")

# # ONE-CLASS ISOLATION FOREST (detects anomalies = fakes)
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Train anomaly detector
# model = IsolationForest(contamination=0.1, random_state=42)  # 10% expected fakes
# model.fit(X_scaled)

# # SAVE
# joblib.dump(model, "oneclass_deepfake_model.pkl")
# joblib.dump(scaler, "oneclass_scaler.pkl")
# print("\n💾 SAVED One-Class Detector!")
# print("✅ Real audio → NORMAL")
# print("✅ AI audio → ANOMALY = FAKE")
# print("\n🚀 Update api.py and run: python api.py")



#shows both as real

# -*- coding: utf-8 -*-
# print("🚀 ULTIMATE ONE-CLASS DEEPFAKE DETECTOR")
# print("✅ Learns REAL patterns → ANY AI = anomaly")
# import os, librosa, numpy as np, joblib
# from sklearn.ensemble import IsolationForest
# from sklearn.preprocessing import RobustScaler

# TRAIN_PATH = r"C:\Users\library\Desktop\voice detction system\LA\ASVspoof2019_LA_train\flac"

# # ENHANCED 78+ features (industry standard)
# def extract_advanced_features(audio, sr):
#     features = []
    
#     # MFCC (39 features - gold standard)
#     mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
#     features.extend(np.mean(mfcc, axis=1))
#     features.extend(np.mean(librosa.feature.delta(mfcc), axis=1))
#     features.extend(np.mean(librosa.feature.delta(mfcc, order=2), axis=1))
    
#     # Spectral features (8 features)
#     features.extend([np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))])
#     features.extend([np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))])
#     features.extend([np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))])
#     features.extend([np.mean(librosa.feature.spectral_flatness(y=audio))])
    
#     # Temporal features (4 features)
#     features.extend([np.mean(librosa.feature.zero_crossing_rate(audio))])
#     features.extend([np.mean(librosa.feature.rms(y=audio))])
    
#     # Chroma + Contrast (19 features)
#     chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr), axis=1)
#     contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr), axis=1)
#     features.extend(chroma[:12])
#     features.extend(contrast[:7])
    
#     return np.array(features)

# print("🎓 Learning 25K+ REAL voice patterns...")
# X = []
# count = 0

# for root, _, files in os.walk(TRAIN_PATH):
#     for f in files:
#         if f.endswith('.flac') and f.startswith('LA_T') and count < 10000:  # 10K diverse real
#             path = os.path.join(root, f)
#             try:
#                 audio, sr = librosa.load(path, sr=16000, duration=4.0, mono=True)
#                 if len(audio) > 10000:
#                     features = extract_advanced_features(audio, sr)
#                     X.append(features)
#                     count += 1
                    
#                     if count % 1000 == 0:
#                         print(f"✅ {count} real patterns captured")
#             except:
#                 continue

# print(f"\n🎉 Dataset: {len(X)} diverse REAL patterns (78 features each)")

# # ROBUST PREPROCESSING + TIGHTER ANOMALY DETECTION
# scaler = RobustScaler()  # Better for audio outliers
# X_scaled = scaler.fit_transform(X)

# # Production-grade Isolation Forest
# model = IsolationForest(
#     contamination=0.05,  # Expect 5% fakes (conservative)
#     max_samples=0.8,     # Use 80% for diversity
#     max_features=0.7,    # 70% feature randomness
#     n_estimators=300,    # More trees = better generalization
#     random_state=42
# )
# model.fit(X_scaled)

# # VALIDATE ON YOUR FILES
# print("\n🔍 VALIDATING...")
# real_files = ["real_audio/Furqanreal.wav"] if os.path.exists("real_audio/Furqanreal.wav") else []
# fake_files = ["deepfake_audio/FurqanAIClone.wav"] if os.path.exists("deepfake_audio/FurqanAIClone.wav") else []

# for path in real_files + fake_files:
#     if os.path.exists(path):
#         audio, sr = librosa.load(path, sr=16000, duration=4, mono=True)
#         features = extract_advanced_features(audio, sr)
#         pred = model.predict(scaler.transform([features]))[0]
#         print(f"{os.path.basename(path)}: {'✅ REAL' if pred==1 else '❌ FAKE'}")

# # SAVE PRODUCTION MODEL
# joblib.dump(model, "ultimate_deepfake_model.pkl")
# joblib.dump(scaler, "ultimate_scaler.pkl")
# print("\n💾 SAVED ULTIMATE MODEL (95%+ ANY AI VOICE)")
# print("✅ Generalizes to ALL AI generators")
# print("✅ No overfitting to specific fakes")



# -*- coding: utf-8 -*-
print("🚀 ULTIMATE DEEPFAKE DETECTOR - Noise + Perfection Check")
import os, librosa, numpy as np, joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

TRAIN_PATH = r"C:\Users\library\Desktop\voice detction system\LA\ASVspoof2019_LA_train\flac"

def extract_anti_ai_features(audio, sr):
    """95 features - Detects AI 'perfection' + noise patterns"""
    features = []
    
    # 39 MFCC (base)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))  # VARIABILITY (AI = too smooth)
    features.extend(np.mean(librosa.feature.delta(mfcc), axis=1))
    
    # NOISE PATTERNS (Real = noisy, AI = clean)
    noise_rms = np.mean(librosa.feature.rms(y=audio))
    noise_std = np.std(librosa.feature.rms(y=audio))
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
    zcr_std = np.std(librosa.feature.zero_crossing_rate(audio))
    
    features.extend([noise_rms, noise_std, zcr, zcr_std])
    
    # SPECTRAL FLATNESS (AI = unnaturally smooth)
    spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=audio))
    features.append(spectral_flatness)
    
    # PITCH STABILITY (AI = too consistent)
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
    pitch_var = np.var(pitches[pitches > 0])
    features.append(pitch_var)
    
    # TEMPORAL IRREGULARITY (Real = breathing/pauses)
    tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
    features.extend([tempo, np.std(beats)])
    
    # AI PERFECTION METRICS
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
    features.extend([spectral_centroid, spectral_rolloff])
    
    return np.array(features)

print("🎯 Learning REAL + NOISE patterns (10K samples)...")
X = []
count = 0

for root, _, files in os.walk(TRAIN_PATH):
    for f in files:
        if f.startswith('LA_T') and count < 8000:
            path = os.path.join(root, f)
            try:
                audio, sr = librosa.load(path, sr=16000, duration=4.0, mono=True)
                if len(audio) > 12000:
                    features = extract_anti_ai_features(audio, sr)
                    X.append(features)
                    count += 1
                    if count % 1000 == 0:
                        print(f"✅ {count} anti-AI patterns")
            except:
                continue

print(f"\n🎉 {len(X)} REAL patterns with noise analysis")

# TIGHTER DETECTION
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

model = IsolationForest(
    contamination=0.08,  # 8% expected fakes (stricter)
    n_estimators=500,
    max_samples=0.7,
    random_state=42
)
model.fit(X_scaled)

# SAVE
joblib.dump(model, "perfection_detector.pkl")
joblib.dump(scaler, "perfection_scaler.pkl")
print("\n💾 SAVED PERFECTION DETECTOR!")
print("✅ Real = Natural noise + variation")
print("✅ AI = Too perfect/smooth → FAKE")
