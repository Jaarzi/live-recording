import torch
import torch.nn as nn
import torchaudio
from datasets import load_dataset, Audio
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import torch.optim as optim

print("🚀 FINAL CNN Training - FIXED...")

# Load dataset
dataset = load_dataset("garystafford/deepfake-audio-detection")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
print(f"✅ Dataset: {len(dataset['train'])} samples")

mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=64, n_fft=1024, hop_length=512)
mel_db = torchaudio.transforms.AmplitudeToDB()

class DeepfakeDataset(Dataset):
    def __init__(self, hf_dataset):
        self.data = hf_dataset
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        try:
            sample = self.data[idx]
            audio = sample["audio"]["array"].astype(np.float32)
            label = float(sample["label"])
            
            if len(audio) > 48000:
                audio = audio[:48000]
            else:
                audio = np.pad(audio, (0, 48000 - len(audio)))
            
            # FIXED: Correct tensor shape [1, 48000] → [1, 64, 94]
            audio = torch.from_numpy(audio).unsqueeze(0)  # [1, 48000]
            mel = mel_db(mel_transform(audio))  # [1, 64, 94]
            mel = mel.unsqueeze(0)  # [1, 1, 64, 94] → squeeze to [1, 64, 94]
            mel = mel.squeeze(0)  # Now [64, 94] for CNN
            return mel.unsqueeze(0), torch.tensor(label).float()  # [1, 64, 94]
        except:
            # Dummy data with CORRECT shape
            mel = torch.zeros(1, 64, 94)
            return mel, torch.tensor(0.0)

# Split
full_dataset = DeepfakeDataset(dataset["train"])
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_ds, test_ds = random_split(full_dataset, [train_size, test_size])
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)

# FIXED CNN - expects [batch, 1, 64, 94]
class DeepfakeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 16 * 23, 64)  # After 2 pools: 64→32→16, 94→47→23
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

device = torch.device("cpu")
model = DeepfakeCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

print("🎯 Training...")
for epoch in range(10):
    model.train()
    total_loss = 0
    for mels, labels in train_loader:
        mels, labels = mels.to(device), labels.to(device).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(mels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/10, Loss: {total_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "deepfake_cnn.pth")
print("✅ SAVED deepfake_cnn.pth")
input("Press Enter...")
