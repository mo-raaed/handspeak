import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import pickle

# Configuration
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_SCRIPT_DIR, os.pardir)

CSV_PATH = os.path.join(_PROJECT_ROOT, "data", "landmarks.csv")
MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "landmark_model.pth")
ENCODER_SAVE_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")

EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a tiny Feed-Forward Neural Network
class LandmarkNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LandmarkNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def load_data():
    X = []
    y = []
    print(f"Loading data from {CSV_PATH}...")
    with open(CSV_PATH, "r") as f:
        reader = csv.reader(f)
        header = next(reader) # skip header
        for row in reader:
            if not row: continue
            label = row[0]
            # Convert the 63 coordinates to floats
            coords = [float(val) for val in row[1:]]
            
            # NORMALIZATION: Make coordinates relative to the wrist (landmark 0)
            wrist_x, wrist_y, wrist_z = coords[0], coords[1], coords[2]
            rel_coords = []
            for i in range(0, len(coords), 3):
                rel_coords.append(coords[i] - wrist_x)
                rel_coords.append(coords[i+1] - wrist_y)
                rel_coords.append(coords[i+2] - wrist_z)
                
            # Normalize by scale (max absolute distance)
            max_val = max([abs(val) for val in rel_coords])
            if max_val > 0:
                rel_coords = [val / max_val for val in rel_coords]
            
            X.append(rel_coords)
            y.append(label)
            
    return np.array(X), np.array(y)

def train():
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # 1. Load Dataset
    X, y = load_data()
    print(f"Loaded {len(X)} samples with {X.shape[1]} features each.")

    # 2. Encode text labels into numbers (A->0, B->1...)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)
    
    # Save the encoder so inference script knows the class names!
    with open(ENCODER_SAVE_PATH, "wb") as f:
        pickle.dump(le, f)
        
    # 3. Train / Val Split
    X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.15, random_state=42)
    
    # Convert to PyTorch tensors
    X_train_t = torch.FloatTensor(X_train).to(DEVICE)
    y_train_t = torch.LongTensor(y_train).to(DEVICE)
    X_val_t = torch.FloatTensor(X_val).to(DEVICE)
    y_val_t = torch.LongTensor(y_val).to(DEVICE)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 4. Build Model
    model = LandmarkNN(input_size=63, num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 5. Training Loop
    best_val_loss = float('inf')
    
    print("Starting fast training...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            _, preds = outputs.max(1)
            train_correct += preds.eq(batch_y).sum().item()
            
        train_loss /= len(X_train)
        train_acc = train_correct / len(X_train)
        
        # Validation Phase
        model.eval()
        with torch.no_grad():
            outputs = model(X_val_t)
            val_loss = criterion(outputs, y_val_t).item()
            _, preds = outputs.max(1)
            val_acc = preds.eq(y_val_t).sum().item() / len(X_val)
            
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            
    print(f"Training finished! Best model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()
