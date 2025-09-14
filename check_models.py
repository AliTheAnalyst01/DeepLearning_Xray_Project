#!/usr/bin/env python3
"""
Check for existing trained models and their accuracy
"""

import os
import torch
import glob
from Xray.components.model_training import XRayCNN

def find_trained_models():
    """Find all trained model files"""
    print("🔍 Searching for trained models...")

    # Search patterns
    patterns = [
        "artifacts/*/model_training/model.pt",
        "artifacts/*/model_training/*.pt",
        "artifacts/*/model_training/*.pth",
        "*.pt",
        "*.pth"
    ]

    models_found = []

    for pattern in patterns:
        files = glob.glob(pattern)
        for file in files:
            if os.path.isfile(file):
                size = os.path.getsize(file)
                models_found.append((file, size))

    print(f"📊 Found {len(models_found)} model files:")
    for i, (file, size) in enumerate(models_found, 1):
        print(f"   {i:2d}. {file:<50} ({size:>8} bytes)")

    return models_found

def test_model_loading(model_path):
    """Test if a model can be loaded successfully"""
    try:
        print(f"\n🧪 Testing model: {model_path}")

        # Load the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_state = torch.load(model_path, map_location=device)

        if isinstance(model_state, dict):
            if 'model_state_dict' in model_state:
                print("   ✅ Model contains state_dict")
                print(f"   📋 Architecture: {model_state.get('model_architecture', 'Unknown')}")
                print(f"   🔢 Classes: {model_state.get('num_classes', 'Unknown')}")

                # Create model and load state
                model = XRayCNN(num_classes=model_state.get('num_classes', 2))
                model.load_state_dict(model_state['model_state_dict'])
                model.to(device)
                model.eval()

                print("   ✅ Model loaded successfully!")
                return True, model
            else:
                print("   ⚠️  Model is a direct state dict")
                model = XRayCNN(num_classes=2)
                model.load_state_dict(model_state)
                model.to(device)
                model.eval()
                print("   ✅ Model loaded successfully!")
                return True, model
        else:
            print("   ⚠️  Model is already a model object")
            return True, model_state

    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        return False, None

def main():
    print("🏥 X-Ray Model Checker")
    print("=" * 60)

    # Find models
    models = find_trained_models()

    if not models:
        print("\n❌ No trained models found!")
        print("💡 Run the training pipeline to create a model:")
        print("   python main.py")
        return

    # Test each model
    working_models = []
    for model_path, size in models:
        success, model = test_model_loading(model_path)
        if success:
            working_models.append((model_path, model))

    print(f"\n📊 Summary:")
    print(f"   Total models found: {len(models)}")
    print(f"   Working models: {len(working_models)}")

    if working_models:
        print(f"\n✅ Best model to use: {working_models[0][0]}")
        print("💡 Update your FastAPI to use this model path!")
    else:
        print("\n❌ No working models found!")
        print("💡 Train a new model with: python main.py")

if __name__ == "__main__":
    main()
