#!/usr/bin/env python3
"""
Download sample X-ray images for testing the model
These images are from a different dataset than what the model was trained on
"""

import requests
import os
from PIL import Image
import io

def download_image(url, filename):
    """Download an image from URL and save it"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Open and save the image
        image = Image.open(io.BytesIO(response.content))
        image.save(filename)
        print(f"✅ Downloaded: {filename}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {filename}: {e}")
        return False

def create_test_images():
    """Create test images by downloading from public sources"""
    
    # Create test_images directory
    os.makedirs("test_images", exist_ok=True)
    
    print("🔄 Downloading test X-ray images...")
    print("These images are from different datasets than your training data")
    print("=" * 60)
    
    # Sample X-ray images from public sources
    # Note: These are placeholder URLs - in practice, you'd use real medical image datasets
    test_images = [
        {
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/Chest_X-ray_after_COVID-19_pneumonia.jpg/800px-Chest_X-ray_after_COVID-19_pneumonia.jpg",
            "filename": "test_images/covid_pneumonia_1.jpg",
            "description": "COVID-19 Pneumonia X-ray"
        },
        {
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Chest_X-ray_of_pneumonia.jpg/800px-Chest_X-ray_of_pneumonia.jpg", 
            "filename": "test_images/pneumonia_1.jpg",
            "description": "Pneumonia X-ray"
        },
        {
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9e/Normal_posteroanterior_chest_X-ray.jpg/800px-Normal_posteroanterior_chest_X-ray.jpg",
            "filename": "test_images/normal_1.jpg", 
            "description": "Normal Chest X-ray"
        }
    ]
    
    downloaded_count = 0
    
    for img in test_images:
        if download_image(img["url"], img["filename"]):
            downloaded_count += 1
            print(f"   📝 {img['description']}")
    
    print("=" * 60)
    print(f"📊 Downloaded {downloaded_count}/{len(test_images)} test images")
    
    # Create some synthetic test images as backup
    create_synthetic_images()
    
    print("\n🎯 Test images are ready in the 'test_images/' directory!")
    print("You can now use these to test your Streamlit app.")

def create_synthetic_images():
    """Create synthetic X-ray-like images for testing"""
    print("\n🔄 Creating synthetic test images...")
    
    try:
        from PIL import Image, ImageDraw, ImageFilter
        import numpy as np
        
        # Create synthetic normal X-ray
        normal_img = Image.new('L', (512, 512), 200)  # Light gray background
        draw = ImageDraw.Draw(normal_img)
        
        # Draw rib cage
        for i in range(10):
            y = 50 + i * 40
            draw.ellipse([100, y, 400, y+20], outline=100, width=2)
        
        # Draw spine
        draw.rectangle([250, 50, 260, 450], fill=80)
        
        # Draw heart shadow
        draw.ellipse([200, 200, 300, 300], outline=120, width=3)
        
        normal_img.save("test_images/synthetic_normal.jpg")
        print("✅ Created: synthetic_normal.jpg")
        
        # Create synthetic pneumonia X-ray
        pneumonia_img = Image.new('L', (512, 512), 200)
        draw = ImageDraw.Draw(pneumonia_img)
        
        # Draw rib cage
        for i in range(10):
            y = 50 + i * 40
            draw.ellipse([100, y, 400, y+20], outline=100, width=2)
        
        # Draw spine
        draw.rectangle([250, 50, 260, 450], fill=80)
        
        # Draw heart shadow
        draw.ellipse([200, 200, 300, 300], outline=120, width=3)
        
        # Add pneumonia-like opacities
        for _ in range(5):
            x = np.random.randint(150, 350)
            y = np.random.randint(150, 350)
            size = np.random.randint(20, 50)
            draw.ellipse([x-size, y-size, x+size, y+size], fill=150)
        
        pneumonia_img.save("test_images/synthetic_pneumonia.jpg")
        print("✅ Created: synthetic_pneumonia.jpg")
        
    except Exception as e:
        print(f"❌ Failed to create synthetic images: {e}")

def main():
    print("🏥 X-Ray Test Image Downloader")
    print("=" * 60)
    print("This script downloads sample X-ray images for testing your model")
    print("These images are from different sources than your training data")
    print("=" * 60)
    
    create_test_images()
    
    print("\n📁 Test Images Directory Contents:")
    if os.path.exists("test_images"):
        for file in os.listdir("test_images"):
            if file.endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join("test_images", file)
                size = os.path.getsize(file_path)
                print(f"   📄 {file} ({size} bytes)")

if __name__ == "__main__":
    main()
