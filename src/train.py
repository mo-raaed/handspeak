import os
import torch
import torch.nn as nn
import torch.optim as optim
from preprocess import get_dataloaders
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm

# Configuration
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_SCRIPT_DIR, os.pardir)
MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")

EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model(num_classes):
    # Use ResNet18 but modify for our number of classes
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    # The default resnet18 takes 224x224, but works fine with 64x64. 
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model.to(DEVICE)

def train():
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    print(f"Loading data...")
    train_loader, val_loader, class_names = get_dataloaders()
    num_classes = len(class_names)
    
    print(f"Building model for {num_classes} classes and moving to {DEVICE}...")
    model = build_model(num_classes)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_loss = float('inf')
    best_model_path = os.path.join(MODELS_DIR, "best_model.pth")
    
    print("Starting training...")
    for epoch in range(EPOCHS):
        # Training Phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for images, labels in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            loop.set_postfix(loss=loss.item())
            
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1} Summary:")
        print(f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f}")
        print(f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"--> Saved new best model to {best_model_path}")
            
if __name__ == "__main__":
    train()
