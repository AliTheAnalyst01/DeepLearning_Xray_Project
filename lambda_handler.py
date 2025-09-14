#!/usr/bin/env python3
"""
AWS Lambda handler for X-Ray Classification API
"""

import json
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import os

# Create FastAPI app
app = FastAPI(title="X-Ray Classification API", version="2.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lightweight model architecture for Lambda
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

# Load the lightweight trained model
def load_model():
    try:
        model_path = "lambda_model.pt"
        device = torch.device("cpu")  # Lambda uses CPU
        
        model_state = torch.load(model_path, map_location=device)
        
        if isinstance(model_state, dict) and 'model_state_dict' in model_state:
            model = LightweightXRayCNN(num_classes=model_state.get('num_classes', 2))
            model.load_state_dict(model_state['model_state_dict'])
        else:
            model = LightweightXRayCNN(num_classes=2)
            model.load_state_dict(model_state)
        
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        print(f"Model loading failed: {e}")
        return None, None

# Load model once
model, device = load_model()

# Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.get("/")
async def root():
    return {"message": "X-Ray Classification API on AWS Lambda!", "status": "healthy"}

@app.get("/health")
async def health():
    return {
        "status": "healthy", 
        "model_loaded": model is not None,
        "model_accuracy": "~90% (Lightweight)",
        "model_architecture": "LightweightXRayCNN",
        "device": str(device) if device else "cpu"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')

        # Preprocess
        input_tensor = transform(image).unsqueeze(0)
        
        if device:
            input_tensor = input_tensor.to(device)

        # Predict
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        # Format results
        class_names = ['NORMAL', 'PNEUMONIA']
        predicted_class = class_names[predicted.item()]
        confidence_score = confidence.item()

        return {
            "prediction": predicted_class,
            "confidence": round(confidence_score, 4),
            "class_probabilities": {
                "NORMAL": round(probabilities[0][0].item(), 4),
                "PNEUMONIA": round(probabilities[0][1].item(), 4)
            },
            "model_info": {
                "architecture": "LightweightXRayCNN",
                "accuracy": "~90% (Optimized for Lambda)",
                "device": str(device) if device else "cpu"
            },
            "status": "success"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

# Create Lambda handler
handler = Mangum(app)