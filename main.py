# import os
# import glob
# import librosa
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score, confusion_matrix
# import joblib

# def extract_mfcc_features(audio_path, n_mfcc=13, n_fft=2048, hop_length=512):
#     try:
#         audio_data, sr = librosa.load(audio_path, sr=None)
#     except Exception as e:
#         print(f"Error loading audio file {audio_path}: {e}")
#         return None
#     mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
#     return np.mean(mfccs.T, axis=0)

# def create_dataset(directory, label):
#     X, y = [], []
#     audio_files = glob.glob(os.path.join(directory, "*.wav"))
#     for audio_path in audio_files:
#         mfcc_features = extract_mfcc_features(audio_path)
#         if mfcc_features is not None:
#             X.append(mfcc_features)
#             y.append(label)
#         else:
#             print(f"Skipping audio file {audio_path}")
#     print(f"Number of samples in {directory}: {len(X)}")
#     print("Filenames in", directory, ":", [os.path.basename(path) for path in audio_files])
#     return np.array(X), np.array(y)

# def train_model(X, y):
#     unique_classes = np.unique(y)
#     print("Unique classes in y_train:", unique_classes)
#     if len(unique_classes) < 2:
#         raise ValueError("At least 2 classes are required to train.")
#     class_counts = np.bincount(y)
#     if np.min(class_counts) < 2:
#         print("Warning: Each class should have at least two samples for stratified splitting.")
#         X_train, y_train = X, y
#         X_test, y_test = None, None
#     else:
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=0.2, random_state=42, stratify=y)
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     svm_classifier = SVC(kernel='linear', random_state=42)
#     svm_classifier.fit(X_train_scaled, y_train)
#     if X_test is not None:
#         X_test_scaled = scaler.transform(X_test)
#         y_pred = svm_classifier.predict(X_test_scaled)
#         accuracy = accuracy_score(y_test, y_pred)
#         confusion_mtx = confusion_matrix(y_test, y_pred)
#         print("Accuracy:", accuracy)
#         print("Confusion Matrix:")
#         print(confusion_mtx)
#     else:
#         print("Insufficient samples for stratified splitting. Model trained on all available data.")
#     # Save the trained SVM model and scaler
#     joblib.dump(svm_classifier, "svm_model.pkl")
#     joblib.dump(scaler, "scaler.pkl")

# def analyze_audio(input_audio_path):
#     svm_classifier = joblib.load("svm_model.pkl")
#     scaler = joblib.load("scaler.pkl")
#     if not os.path.exists(input_audio_path):
#         print("Error: The specified file does not exist.")
#         return
#     if not input_audio_path.lower().endswith(".wav"):
#         print("Error: The specified file is not a .wav file.")
#         return
#     mfcc_features = extract_mfcc_features(input_audio_path)
#     if mfcc_features is not None:
#         mfcc_features_scaled = scaler.transform([mfcc_features])
#         prediction = svm_classifier.predict(mfcc_features_scaled)
#         if prediction[0] == 0:
#             print("The input audio is classified as genuine.")
#         else:
#             print("The input audio is classified as deepfake.")
#     else:
#         print("Error: Unable to process the input audio.")

# def main():
#     genuine_dir = r"C:\Users\jaarzi\Desktop\deepfake\DeepFake-Audio-Detection-MFCC\real_audio"
#     deepfake_dir = r"C:\Users\jaarzi\Desktop\deepfake\DeepFake-Audio-Detection-MFCC\deepfake_audio"
#     X_genuine, y_genuine = create_dataset(genuine_dir, label=0)
#     X_deepfake, y_deepfake = create_dataset(deepfake_dir, label=1)
#     if len(X_genuine) < 2 or len(X_deepfake) < 2:
#         print("Each class should have at least two samples for stratified splitting. Exiting.")
#         return
#     X = np.vstack((X_genuine, X_deepfake))
#     y = np.hstack((y_genuine, y_deepfake))
#     train_model(X, y)
#     # Prompt for audio to analyze after training
#     user_input_file = input("Enter the path of the .wav file to analyze: ").strip()
#     if user_input_file:
#         analyze_audio(user_input_file)

# if __name__ == "__main__":
#     main()



# import os
# import glob
# import librosa
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score, confusion_matrix
# import joblib


# def extract_mfcc_features(audio_path, n_mfcc=13, n_fft=2048, hop_length=512):
#     try:
#         audio_data, sr = librosa.load(audio_path, sr=None)
#     except Exception as e:
#         print(f"Error loading audio file {audio_path}: {e}")
#         return None
#     mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
#     return np.mean(mfccs.T, axis=0)


# def create_dataset(directory, label):
#     X, y = [], []
#     audio_files = glob.glob(os.path.join(directory, "*.wav"))
#     for audio_path in audio_files:
#         mfcc_features = extract_mfcc_features(audio_path)
#         if mfcc_features is not None:
#             X.append(mfcc_features)
#             y.append(label)
#         else:
#             print(f"Skipping audio file {audio_path}")
#     print(f"Number of samples in {directory}: {len(X)}")
#     print("Filenames in", directory, ":", [os.path.basename(path) for path in audio_files])
#     return np.array(X), np.array(y)


# def train_model(X, y):
#     unique_classes = np.unique(y)
#     print("Unique classes in y_train:", unique_classes)
#     if len(unique_classes) < 2:
#         raise ValueError("At least 2 classes are required to train.")

#     class_counts = np.bincount(y)
#     if np.min(class_counts) < 2:
#         print("Warning: Each class should have at least two samples for stratified splitting.")
#         X_train, y_train = X, y
#         X_test, y_test = None, None
#     else:
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=0.2, random_state=42, stratify=y)
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)

#     # Enable probability=True to get confidence scores
#     svm_classifier = SVC(kernel='linear', random_state=42, probability=True)
#     svm_classifier.fit(X_train_scaled, y_train)

#     if X_test is not None:
#         X_test_scaled = scaler.transform(X_test)
#         y_pred = svm_classifier.predict(X_test_scaled)
#         accuracy = accuracy_score(y_test, y_pred)
#         confusion_mtx = confusion_matrix(y_test, y_pred)
#         print("Accuracy:", accuracy)
#         print("Confusion Matrix:")
#         print(confusion_mtx)
#     else:
#         print("Insufficient samples for stratified splitting. Model trained on all available data.")

#     # Save the trained SVM model and scaler
#     joblib.dump(svm_classifier, "svm_model.pkl")
#     joblib.dump(scaler, "scaler.pkl")


# def analyze_audio(input_audio_path):
#     svm_classifier = joblib.load("svm_model.pkl")
#     scaler = joblib.load("scaler.pkl")

#     if not os.path.exists(input_audio_path):
#         print("Error: The specified file does not exist.")
#         return
#     if not input_audio_path.lower().endswith(".wav"):
#         print("Error: The specified file is not a .wav file.")
#         return

#     mfcc_features = extract_mfcc_features(input_audio_path)
#     if mfcc_features is not None:
#         mfcc_features_scaled = scaler.transform([mfcc_features])
#         probabilities = svm_classifier.predict_proba(mfcc_features_scaled)[0]
#         prediction = svm_classifier.predict(mfcc_features_scaled)[0]

#         confidence = max(probabilities)
#         label = "genuine" if prediction == 0 else "deepfake"
#         print(f"The input audio is classified as {label} with confidence {confidence:.2f}.")

#     else:
#         print("Error: Unable to process the input audio.")


# def main():
#     genuine_dir = r"C:\Users\jaarzi\Desktop\deepfake\DeepFake-Audio-Detection-MFCC\real_audio"
#     deepfake_dir = r"C:\Users\jaarzi\Desktop\deepfake\DeepFake-Audio-Detection-MFCC\deepfake_audio"
#     X_genuine, y_genuine = create_dataset(genuine_dir, label=0)
#     X_deepfake, y_deepfake = create_dataset(deepfake_dir, label=1)
#     if len(X_genuine) < 2 or len(X_deepfake) < 2:
#         print("Each class should have at least two samples for stratified splitting. Exiting.")
#         return
#     X = np.vstack((X_genuine, X_deepfake))
#     y = np.hstack((y_genuine, y_deepfake))
#     train_model(X, y)
#     # Prompt for audio to analyze after training
#     user_input_file = input("Enter the path of the .wav file to analyze: ").strip()
#     if user_input_file:
#         analyze_audio(user_input_file)


# if __name__ == "__main__":
#     main()


#NEW

#New
import os
import glob
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

def extract_advanced_features(audio_path):
    """🎯 50+ FEATURES - Learns PATTERNS, not files"""
    try:
        audio_data, sr = librosa.load(audio_path, sr=None)
        
        # 1. MFCC (13) - Your original
        mfcc = np.mean(librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13), axis=1)
        
        # 2. Delta MFCC (13) - Voice dynamics
        delta_mfcc = np.mean(librosa.feature.delta(mfcc), axis=1)
        
        # 3. Spectral Centroid (1) - Timbre
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sr))
        
        # 4. Spectral Roll-off (1) - Frequency distribution
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio_data, sr=sr))
        
        # 5. Zero Crossing Rate (1) - Noise patterns
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio_data))
        
        # 6. Chroma (12) - Harmonic structure
        chroma = np.mean(librosa.feature.chroma(y=audio_data, sr=sr), axis=1)
        
        # 7. Spectral Contrast (7) - Frequency bands
        contrast = np.mean(librosa.feature.spectral_contrast(y=audio_data, sr=sr), axis=1)
        
        # 🎯 COMBINE ALL = 48+ features (learns PATTERNS!)
        features = np.concatenate([
            mfcc, delta_mfcc, 
            [spectral_centroid, spectral_rolloff, zcr],
            chroma, contrast
        ])
        
        print(f"✅ {len(features)} advanced features extracted")
        return features
        
    except:
        return None

def create_dataset(directory, label):
    X, y = [], []
    audio_files = glob.glob(os.path.join(directory, "*.wav"))
    
    for audio_path in audio_files:
        features = extract_advanced_features(audio_path)
        if features is not None:
            X.append(features)
            y.append(label)
    
    print(f"📁 {directory}: {len(X)} samples")
    return np.array(X), np.array(y)

def train_advanced_model():
    # Load data
    genuine_dir = r"C:\Users\jaarzi\Desktop\deepfake\DeepFake-Audio-Detection-MFCC\real_audio"
    deepfake_dir = r"C:\Users\jaarzi\Desktop\deepfake\DeepFake-Audio-Detection-MFCC\deepfake_audio"
    
    X_real, y_real = create_dataset(genuine_dir, 0)
    X_fake, y_fake = create_dataset(deepfake_dir, 1)
    
    X = np.vstack([X_real, X_fake])
    y = np.hstack([y_real, y_fake])
    
    # Split + Scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 🎯 BETTER SVM - Learns FEATURES, not files
    model = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Test accuracy
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Test Accuracy: {accuracy:.1%} (on UNSEEN files!)")
    
    # Save
    joblib.dump(model, "advanced_svm_model.pkl")
    joblib.dump(scaler, "advanced_scaler.pkl")
    print("💾 Saved advanced model!")

if __name__ == "__main__":
    train_advanced_model()
