#!/usr/bin/env python3
"""
Test script to verify local API and Streamlit setup
"""

import requests
import time
from PIL import Image
import io

def test_api():
    """Test the FastAPI backend"""
    print("🧪 Testing FastAPI Backend...")

    try:
        # Test health endpoint
        response = requests.get("http://localhost:8001/health", timeout=5)
        if response.status_code == 200:
            print("✅ API Health Check: PASSED")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ API Health Check: FAILED ({response.status_code})")
            return False
    except Exception as e:
        print(f"❌ API Health Check: ERROR - {e}")
        return False

    # Test prediction endpoint
    try:
        # Create a test image
        test_image = Image.new('RGB', (224, 224), color='white')
        img_byte_arr = io.BytesIO()
        test_image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        # Send prediction request
        files = {'file': ('test.jpg', img_byte_arr, 'image/jpeg')}
        response = requests.post("http://localhost:8001/predict", files=files, timeout=10)

        if response.status_code == 200:
            result = response.json()
            print("✅ API Prediction: PASSED")
            print(f"   Prediction: {result['prediction']}")
            print(f"   Confidence: {result['confidence']:.2%}")
            return True
        else:
            print(f"❌ API Prediction: FAILED ({response.status_code})")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ API Prediction: ERROR - {e}")
        return False

def test_streamlit():
    """Test if Streamlit is accessible"""
    print("\n🧪 Testing Streamlit Frontend...")

    try:
        response = requests.get("http://localhost:8501", timeout=5)
        if response.status_code == 200:
            print("✅ Streamlit Frontend: ACCESSIBLE")
            return True
        else:
            print(f"❌ Streamlit Frontend: NOT ACCESSIBLE ({response.status_code})")
            return False
    except Exception as e:
        print(f"❌ Streamlit Frontend: ERROR - {e}")
        return False

def main():
    print("🚀 Testing Local X-Ray Classification Setup")
    print("=" * 50)

    # Test API
    api_ok = test_api()

    # Test Streamlit
    streamlit_ok = test_streamlit()

    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)

    if api_ok:
        print("✅ FastAPI Backend: WORKING")
        print("   🌐 URL: http://localhost:8001")
        print("   📊 Health: http://localhost:8001/health")
        print("   🔍 Predict: http://localhost:8001/predict")
        print("   📚 Docs: http://localhost:8001/docs")
    else:
        print("❌ FastAPI Backend: NOT WORKING")

    if streamlit_ok:
        print("✅ Streamlit Frontend: WORKING")
        print("   🌐 URL: http://localhost:8501")
    else:
        print("❌ Streamlit Frontend: NOT WORKING")
        print("   💡 Try: streamlit run streamlit_app.py --server.port 8501")

    print("\n🎯 NEXT STEPS:")
    if api_ok and streamlit_ok:
        print("1. Open your browser and go to: http://localhost:8501")
        print("2. Upload an X-ray image to test the prediction")
        print("3. Watch the AI analyze your image in real-time!")
    elif api_ok:
        print("1. Start Streamlit: streamlit run streamlit_app.py --server.port 8501")
        print("2. Then open: http://localhost:8501")
    else:
        print("1. Start API: python3 fast_api.py")
        print("2. Start Streamlit: streamlit run streamlit_app.py --server.port 8501")
        print("3. Open: http://localhost:8501")

if __name__ == "__main__":
    main()
