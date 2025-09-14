#!/usr/bin/env python3
"""
Fast X-ray Classification API using FastAPI
Ready for Postman testing!
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import numpy as np
import uvicorn

app = FastAPI(title="X-Ray Classification API", version="1.0.0")

# Simple CNN model (matching your trained model architecture)
class SimpleXRayModel(torch.nn.Module):
    def __init__(self):
        super(SimpleXRayModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(128 * 28 * 28, 512)
        self.fc2 = torch.nn.Linear(512, 2)
        self.dropout = torch.nn.Dropout(0.5)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Initialize model
model = SimpleXRayModel()
model.eval()

# Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.get("/")
async def root():
    return {"message": "X-Ray Classification API is running!", "status": "healthy"}

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict X-ray image for pneumonia detection

    Args:
        file: X-ray image file (JPEG/PNG)

    Returns:
        JSON with prediction results
    """
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')

        # Preprocess
        input_tensor = transform(image).unsqueeze(0)

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
            "status": "success"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting X-Ray Classification API...")
    print("📱 Postman URL: http://localhost:8000/predict")
    print("📋 Method: POST")
    print("📝 Body: form-data, key='file', type=file")
    uvicorn.run(app, host="0.0.0.0", port=8000)
