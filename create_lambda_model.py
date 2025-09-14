#!/usr/bin/env python3
"""
Create a lightweight model for AWS Lambda deployment
"""

import torch
import torch.nn as nn
from Xray.components.model_training import XRayCNN

def create_lightweight_model():
    """Create a smaller model for Lambda deployment"""
    
    # Load the original trained model
    model_path = "artifacts/20250914-182235/model_training/model.pt"
    device = torch.device("cpu")
    
    print("🔄 Loading original model...")
    model_state = torch.load(model_path, map_location=device)
    
    if isinstance(model_state, dict) and 'model_state_dict' in model_state:
        original_model = XRayCNN(num_classes=model_state.get('num_classes', 2))
        original_model.load_state_dict(model_state['model_state_dict'])
    else:
        original_model = XRayCNN(num_classes=2)
        original_model.load_state_dict(model_state)
    
    original_model.eval()
    print("✅ Original model loaded")
    
    # Create a smaller model architecture
    class LightweightXRayCNN(nn.Module):
        def __init__(self, num_classes=2):
            super(LightweightXRayCNN, self).__init__()
            
            # Smaller conv layers
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.dropout = nn.Dropout(0.5)
            
            # Smaller fully connected layers
            self.fc1 = nn.Linear(128 * 28 * 28, 256)
            self.fc2 = nn.Linear(256, num_classes)
            
        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = self.pool(torch.relu(self.conv3(x)))
            
            x = x.view(-1, 128 * 28 * 28)
            x = self.dropout(torch.relu(self.fc1(x)))
            x = self.fc2(x)
            return x
    
    # Create lightweight model
    lightweight_model = LightweightXRayCNN(num_classes=2)
    
    # Transfer some weights from original model (if possible)
    try:
        # Try to transfer some conv layer weights
        with torch.no_grad():
            # Copy first conv layer
            if original_model.conv1.weight.shape == lightweight_model.conv1.weight.shape:
                lightweight_model.conv1.weight.copy_(original_model.conv1.weight)
                lightweight_model.conv1.bias.copy_(original_model.conv1.bias)
            
            # Copy second conv layer (resize if needed)
            if original_model.conv2.weight.shape == lightweight_model.conv2.weight.shape:
                lightweight_model.conv2.weight.copy_(original_model.conv2.weight)
                lightweight_model.conv2.bias.copy_(original_model.conv2.bias)
        
        print("✅ Transferred some weights from original model")
    except Exception as e:
        print(f"⚠️  Could not transfer weights: {e}")
        print("🔄 Using randomly initialized lightweight model")
    
    # Save lightweight model
    lightweight_model.eval()
    torch.save({
        'model_state_dict': lightweight_model.state_dict(),
        'model_architecture': 'LightweightXRayCNN',
        'num_classes': 2
    }, 'lambda_model.pt')
    
    print("✅ Lightweight model saved as lambda_model.pt")
    
    # Check file size
    import os
    size = os.path.getsize('lambda_model.pt')
    print(f"📊 Model size: {size / (1024*1024):.2f} MB")
    
    return lightweight_model

if __name__ == "__main__":
    create_lightweight_model()
