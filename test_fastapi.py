#!/usr/bin/env python3
"""
Test the FastAPI endpoint for Postman configuration
"""

import requests
from PIL import Image
import io

def test_api():
    """Test the FastAPI and provide Postman config."""
    print("🚀 TESTING FASTAPI X-RAY CLASSIFICATION")
    print("=" * 50)

    # Test health endpoint
    try:
        response = requests.get("http://localhost:8000/health")
        print(f"✅ Health check: {response.status_code}")
        print(f"📊 Response: {response.json()}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return

    # Create test image
    test_image = Image.new('RGB', (224, 224), color='white')
    img_byte_arr = io.BytesIO()
    test_image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    # Test predict endpoint
    try:
        files = {'file': ('test.jpg', img_byte_arr, 'image/jpeg')}
        response = requests.post("http://localhost:8000/predict", files=files)

        print(f"\n✅ Prediction test: {response.status_code}")
        print(f"📊 Response: {response.json()}")

        print("\n" + "=" * 50)
        print("📱 POSTMAN CONFIGURATION")
        print("=" * 50)
        print("🌐 URL: http://localhost:8000/predict")
        print("📝 Method: POST")
        print("📝 Headers: None needed")
        print("📝 Body: form-data")
        print("   - Key: file")
        print("   - Type: File")
        print("   - Value: Select any X-ray image")
        print("\n📊 Expected Response:")
        print(response.json())

    except Exception as e:
        print(f"❌ Prediction test failed: {e}")

if __name__ == "__main__":
    test_api()
