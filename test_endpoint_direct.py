#!/usr/bin/env python3
"""
Direct test of the API endpoint to verify it's working.
"""

import requests
import json
from PIL import Image
import io
import time

def create_test_image():
    """Create a simple test image."""
    # Create a white test image
    img = Image.new('RGB', (224, 224), color='white')

    # Convert to bytes
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    return img_byte_arr.getvalue()

def test_endpoints():
    """Test various endpoints to find the correct one."""
    base_url = "http://localhost:3000"

    print("🔍 Testing available endpoints...")

    # Test common endpoints
    endpoints = [
        "/",
        "/healthz",
        "/health",
        "/predict",
        "/api/predict",
        "/v1/predict",
        "/docs",
        "/openapi.json"
    ]

    for endpoint in endpoints:
        try:
            url = base_url + endpoint
            print(f"\n🌐 Testing: {url}")

            if endpoint == "/predict":
                # For predict endpoint, send image data
                img_data = create_test_image()
                response = requests.post(url, data=img_data,
                                       headers={"Content-Type": "image/jpeg"},
                                       timeout=10)
            else:
                # For other endpoints, use GET
                response = requests.get(url, timeout=5)

            print(f"   Status: {response.status_code}")
            print(f"   Content-Type: {response.headers.get('content-type', 'N/A')}")

            if response.status_code == 200:
                if 'json' in response.headers.get('content-type', ''):
                    try:
                        data = response.json()
                        print(f"   Response: {json.dumps(data, indent=2)[:200]}...")
                    except:
                        print(f"   Response: {response.text[:200]}...")
                else:
                    print(f"   Response: {response.text[:200]}...")

        except requests.exceptions.Timeout:
            print(f"   ⏰ Timeout")
        except requests.exceptions.ConnectionError:
            print(f"   ❌ Connection Error")
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")

def test_with_real_image():
    """Test with a real X-ray image from the dataset."""
    print("\n" + "=" * 60)
    print("🩺 TESTING WITH REAL X-RAY IMAGE")
    print("=" * 60)

    # Find a real X-ray image
    import os
    test_data_dir = "artifacts/20250914-143644/data_ingestion/data/test"

    if os.path.exists(test_data_dir):
        # Try to find a NORMAL image
        normal_dir = os.path.join(test_data_dir, "NORMAL")
        if os.path.exists(normal_dir):
            images = [f for f in os.listdir(normal_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if images:
                image_path = os.path.join(normal_dir, images[0])
                print(f"📁 Using image: {image_path}")

                try:
                    with open(image_path, 'rb') as f:
                        img_data = f.read()

                    print(f"📤 Image size: {len(img_data)} bytes")

                    # Test the predict endpoint
                    url = "http://localhost:3000/predict"
                    response = requests.post(url, data=img_data,
                                           headers={"Content-Type": "image/jpeg"},
                                           timeout=30)

                    print(f"📊 Status: {response.status_code}")
                    print(f"📊 Content-Type: {response.headers.get('content-type', 'N/A')}")

                    if response.status_code == 200:
                        try:
                            result = response.json()
                            print("✅ SUCCESS! API Response:")
                            print(json.dumps(result, indent=2))

                            print("\n" + "=" * 60)
                            print("🎯 POSTMAN CONFIGURATION")
                            print("=" * 60)
                            print("✅ Your API is working!")
                            print(f"🌐 URL: {url}")
                            print("📝 Method: POST")
                            print("📝 Headers: Content-Type: image/jpeg")
                            print("📝 Body: binary (select any X-ray image)")
                            print("📝 Expected Response:")
                            print(json.dumps(result, indent=2))

                            return True
                        except Exception as e:
                            print(f"❌ JSON parsing error: {e}")
                            print(f"Raw response: {response.text}")
                    else:
                        print(f"❌ Error: {response.status_code}")
                        print(f"Response: {response.text}")

                except Exception as e:
                    print(f"❌ Error testing with real image: {e}")
            else:
                print("❌ No images found in NORMAL directory")
        else:
            print("❌ NORMAL directory not found")
    else:
        print("❌ Test data directory not found")

    return False

if __name__ == "__main__":
    print("🧪 TESTING X-RAY CLASSIFICATION API ENDPOINTS")
    print("=" * 60)

    # Test all endpoints
    test_endpoints()

    # Test with real image
    success = test_with_real_image()

    if success:
        print("\n🎉 Your API is ready for Postman testing!")
    else:
        print("\n❌ API testing failed. Check service status.")
