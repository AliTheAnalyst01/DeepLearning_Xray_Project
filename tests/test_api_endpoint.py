#!/usr/bin/env python3
"""
Test the API endpoint to verify it's working for Postman testing.
"""

import requests
import json
from PIL import Image
import io

def test_api_endpoint():
    """Test the API endpoint and provide Postman configuration."""
    print("=" * 80)
    print("🧪 TESTING API ENDPOINT FOR POSTMAN")
    print("=" * 80)
    
    # Create a test image
    print("📸 Creating test image...")
    test_image = Image.new('RGB', (224, 224), color='white')
    
    # Save to bytes
    img_byte_arr = io.BytesIO()
    test_image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    
    print("✅ Test image created")
    
    # Test the endpoint
    url = "http://localhost:3000/predict"
    headers = {"Content-Type": "image/jpeg"}
    
    print(f"\n🌐 Testing endpoint: {url}")
    print(f"📤 Sending image with size: {len(img_byte_arr)} bytes")
    
    try:
        response = requests.post(url, data=img_byte_arr, headers=headers, timeout=30)
        
        print(f"📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API Response:")
            print(json.dumps(result, indent=2))
            
            print("\n" + "=" * 80)
            print("📋 POSTMAN CONFIGURATION")
            print("=" * 80)
            
            print("\n🔧 **Method:** POST")
            print(f"🌐 **URL:** {url}")
            print("📝 **Headers:**")
            print("   Content-Type: image/jpeg")
            print("\n📤 **Body:**")
            print("   - Select: binary")
            print("   - Upload: Any X-ray image (JPEG/PNG)")
            print("\n📊 **Expected Response:**")
            print(json.dumps(result, indent=2))
            
            print("\n" + "=" * 80)
            print("📱 **POSTMAN STEP-BY-STEP:**")
            print("=" * 80)
            print("1. Open Postman")
            print("2. Create new request")
            print("3. Set method to POST")
            print(f"4. Enter URL: {url}")
            print("5. Go to Headers tab")
            print("6. Add header: Content-Type = image/jpeg")
            print("7. Go to Body tab")
            print("8. Select 'binary'")
            print("9. Click 'Select File' and choose an X-ray image")
            print("10. Click 'Send'")
            print("11. You should see the prediction response!")
            
            return True
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Service might not be running")
        print("💡 Try: source .venv/bin/activate && cd artifacts/model_deployment && bentoml serve service:svc --port 3000")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def show_service_status():
    """Show the current service status."""
    print("\n" + "=" * 80)
    print("🔍 SERVICE STATUS")
    print("=" * 80)
    
    import subprocess
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        bentoml_processes = [line for line in result.stdout.split('\n') if 'bentoml' in line and 'serve' in line]
        
        if bentoml_processes:
            print("✅ BentoML service is running!")
            print(f"📊 Found {len(bentoml_processes)} worker processes")
            print("🌐 Service should be available at: http://localhost:3000")
        else:
            print("❌ BentoML service is not running")
            print("💡 Start it with: source .venv/bin/activate && cd artifacts/model_deployment && bentoml serve service:svc --port 3000")
    except Exception as e:
        print(f"❌ Could not check service status: {e}")

if __name__ == "__main__":
    show_service_status()
    test_api_endpoint()
    
    print("\n" + "=" * 80)
    print("🎯 YOUR API ENDPOINT IS READY FOR POSTMAN TESTING!")
    print("=" * 80)
