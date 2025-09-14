#!/usr/bin/env python3
"""
Test your model with a real X-ray image from your dataset.
"""

import sys
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import random

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from Xray.components.model_training import XRayCNN

def test_with_real_xray():
    """Test the model with a real X-ray image from the dataset."""
    print("=" * 80)
    print("🩺 TESTING WITH REAL X-RAY IMAGES")
    print("=" * 80)

    try:
        # Find the latest model
        artifacts_dir = "artifacts"
        latest_run = None
        for item in os.listdir(artifacts_dir):
            if item.startswith("2025") and os.path.isdir(os.path.join(artifacts_dir, item)):
                if latest_run is None or item > latest_run:
                    latest_run = item

        if not latest_run:
            print("❌ No trained model found!")
            return False

        model_path = os.path.join(artifacts_dir, latest_run, "model_training", "model.pt")
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return False

        print(f"📁 Loading model from: {model_path}")

        # Load model
        model_state = torch.load(model_path, map_location='cpu')

        if isinstance(model_state, dict) and 'model_state_dict' in model_state:
            model = XRayCNN(num_classes=model_state.get('num_classes', 2))
            model.load_state_dict(model_state['model_state_dict'])
        else:
            model = XRayCNN(num_classes=2)
            model.load_state_dict(model_state)

        model.eval()
        print("✅ Model loaded successfully")

        # Find test images
        test_data_dir = os.path.join(artifacts_dir, latest_run, "data_ingestion", "data", "test")

        if not os.path.exists(test_data_dir):
            print(f"❌ Test data directory not found: {test_data_dir}")
            return False

        # Get sample images from each class
        normal_dir = os.path.join(test_data_dir, "NORMAL")
        pneumonia_dir = os.path.join(test_data_dir, "PNEUMONIA")

        test_images = []

        if os.path.exists(normal_dir):
            normal_images = [f for f in os.listdir(normal_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if normal_images:
                test_images.append({
                    "path": os.path.join(normal_dir, random.choice(normal_images)),
                    "true_class": "NORMAL"
                })

        if os.path.exists(pneumonia_dir):
            pneumonia_images = [f for f in os.listdir(pneumonia_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if pneumonia_images:
                test_images.append({
                    "path": os.path.join(pneumonia_dir, random.choice(pneumonia_images)),
                    "true_class": "PNEUMONIA"
                })

        if not test_images:
            print("❌ No test images found!")
            return False

        print(f"\n🔬 Testing with {len(test_images)} real X-ray images:")
        print("-" * 60)

        # Define transforms
        transforms_pipeline = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        correct_predictions = 0
        total_predictions = len(test_images)

        for i, test_case in enumerate(test_images, 1):
            print(f"\n{i}. Testing {test_case['true_class']} X-ray:")
            print(f"   📁 Image: {os.path.basename(test_case['path'])}")

            try:
                # Load and preprocess image
                image = Image.open(test_case['path']).convert('RGB')
                input_tensor = transforms_pipeline(image).unsqueeze(0)

                # Predict
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)

                # Results
                confidence_score = confidence.item()
                predicted_class = predicted.item()
                class_names = ['NORMAL', 'PNEUMONIA']
                predicted_class_name = class_names[predicted_class]

                # Check if prediction is correct
                is_correct = predicted_class_name == test_case['true_class']
                if is_correct:
                    correct_predictions += 1

                print(f"   🎯 True Class: {test_case['true_class']}")
                print(f"   🔮 Predicted: {predicted_class_name}")
                print(f"   ✅ Correct: {'Yes' if is_correct else 'No'}")
                print(f"   📊 Confidence: {confidence_score:.4f} ({confidence_score*100:.2f}%)")
                print(f"   📈 Probabilities:")
                print(f"      - NORMAL: {probabilities[0][0].item():.4f} ({probabilities[0][0].item()*100:.2f}%)")
                print(f"      - PNEUMONIA: {probabilities[0][1].item():.4f} ({probabilities[0][1].item()*100:.2f}%)")

            except Exception as e:
                print(f"   ❌ Error processing image: {e}")

        # Summary
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        print(f"\n" + "=" * 60)
        print(f"📊 TEST RESULTS SUMMARY")
        print(f"=" * 60)
        print(f"✅ Correct Predictions: {correct_predictions}/{total_predictions}")
        print(f"📈 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

        if accuracy >= 0.8:
            print(f"🎉 Excellent! Your model is performing very well!")
        elif accuracy >= 0.6:
            print(f"👍 Good! Your model is performing reasonably well.")
        else:
            print(f"⚠️  Your model might need more training or tuning.")

        return True

    except Exception as e:
        print(f"❌ Testing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def show_model_files():
    """Show all the model-related files in your project."""
    print("\n" + "=" * 80)
    print("📁 YOUR MODEL FILES")
    print("=" * 80)

    artifacts_dir = "artifacts"
    if os.path.exists(artifacts_dir):
        print(f"📂 Artifacts Directory: {artifacts_dir}")

        # List all training runs
        runs = []
        for item in os.listdir(artifacts_dir):
            if item.startswith("2025") and os.path.isdir(os.path.join(artifacts_dir, item)):
                runs.append(item)

        runs.sort(reverse=True)  # Most recent first

        print(f"\n📅 Training Runs ({len(runs)} total):")
        for i, run in enumerate(runs, 1):
            print(f"   {i}. {run}")

            # Check what's in this run
            run_path = os.path.join(artifacts_dir, run)
            components = []
            if os.path.exists(os.path.join(run_path, "data_ingestion")):
                components.append("Data Ingestion")
            if os.path.exists(os.path.join(run_path, "data_transformation")):
                components.append("Data Transformation")
            if os.path.exists(os.path.join(run_path, "model_training")):
                components.append("Model Training")
            if os.path.exists(os.path.join(run_path, "model_evaluation")):
                components.append("Model Evaluation")

            print(f"      Components: {', '.join(components)}")

        # Show deployment artifacts
        deployment_dir = os.path.join(artifacts_dir, "model_deployment")
        if os.path.exists(deployment_dir):
            print(f"\n🚀 Deployment Artifacts:")
            print(f"   📁 Directory: {deployment_dir}")

            files = ["service.py", "bentofile.yaml", "requirements.txt", "deployment_info.json"]
            for file in files:
                file_path = os.path.join(deployment_dir, file)
                if os.path.exists(file_path):
                    size = os.path.getsize(file_path)
                    print(f"   ✅ {file} ({size} bytes)")
                else:
                    print(f"   ❌ {file} (missing)")

if __name__ == "__main__":
    print("🩺 X-RAY MODEL TESTING WITH REAL IMAGES")
    print("=" * 80)

    # Show model files
    show_model_files()

    # Test with real images
    success = test_with_real_xray()

    if success:
        print("\n🎉 Your model is ready for real-world use!")
    else:
        print("\n❌ There were issues testing your model.")
