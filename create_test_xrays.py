#!/usr/bin/env python3
"""
Create realistic test X-ray images for model testing
These images simulate real X-ray characteristics but are synthetic
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import os
import random

def create_normal_xray(filename, size=(512, 512)):
    """Create a synthetic normal chest X-ray"""
    # Create base image with gradient
    img = Image.new('L', size, 200)
    draw = ImageDraw.Draw(img)

    # Add gradient background
    for y in range(size[1]):
        intensity = int(180 + (y / size[1]) * 40)  # Gradient from 180 to 220
        draw.line([(0, y), (size[0], y)], fill=intensity)

    # Draw rib cage (elliptical shapes)
    for i in range(12):
        y_pos = 80 + i * 30
        width = 300 - i * 8  # Ribs get narrower toward bottom
        height = 15 + random.randint(-3, 3)
        x_center = size[0] // 2

        # Left rib
        draw.ellipse([x_center - width//2, y_pos, x_center - width//2 + width, y_pos + height],
                    outline=120, width=2)
        # Right rib
        draw.ellipse([x_center + width//2 - width, y_pos, x_center + width//2, y_pos + height],
                    outline=120, width=2)

    # Draw spine (vertical line in center)
    spine_width = 15
    draw.rectangle([size[0]//2 - spine_width//2, 50, size[0]//2 + spine_width//2, size[1] - 50],
                  fill=80)

    # Draw heart shadow (left side)
    heart_x = size[0]//2 - 80
    heart_y = 200
    heart_width = 100
    heart_height = 120
    draw.ellipse([heart_x, heart_y, heart_x + heart_width, heart_y + heart_height],
                outline=110, width=3)

    # Draw lungs (darker areas on sides)
    # Left lung
    draw.ellipse([50, 100, size[0]//2 - 20, 400], outline=90, width=2)
    # Right lung
    draw.ellipse([size[0]//2 + 20, 100, size[0] - 50, 400], outline=90, width=2)

    # Add some lung texture
    for _ in range(20):
        x = random.randint(60, size[0]//2 - 30)
        y = random.randint(120, 380)
        size_dot = random.randint(2, 5)
        draw.ellipse([x, y, x + size_dot, y + size_dot], fill=95)

    for _ in range(20):
        x = random.randint(size[0]//2 + 30, size[0] - 60)
        y = random.randint(120, 380)
        size_dot = random.randint(2, 5)
        draw.ellipse([x, y, x + size_dot, y + size_dot], fill=95)

    # Add some noise for realism
    img_array = np.array(img)
    noise = np.random.normal(0, 5, img_array.shape)
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array)

    # Apply slight blur for realism
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

    img.save(filename)
    return img

def create_pneumonia_xray(filename, size=(512, 512)):
    """Create a synthetic pneumonia X-ray"""
    # Start with normal X-ray
    img = create_normal_xray("temp_normal.jpg", size)
    draw = ImageDraw.Draw(img)

    # Add pneumonia opacities (white patches)
    num_opacities = random.randint(3, 8)

    for _ in range(num_opacities):
        # Random position in lung areas
        x = random.randint(60, size[0] - 60)
        y = random.randint(120, 400)

        # Create irregular opacity
        opacity_size = random.randint(20, 60)
        opacity_intensity = random.randint(180, 255)

        # Create irregular shape
        points = []
        for angle in range(0, 360, 30):
            radius = opacity_size + random.randint(-10, 10)
            px = x + radius * np.cos(np.radians(angle))
            py = y + radius * np.sin(np.radians(angle))
            points.append((px, py))

        if len(points) >= 3:
            draw.polygon(points, fill=opacity_intensity)

    # Add some consolidation areas (larger white patches)
    for _ in range(random.randint(1, 3)):
        x = random.randint(80, size[0] - 80)
        y = random.randint(150, 350)
        width = random.randint(40, 80)
        height = random.randint(30, 60)

        # Create irregular rectangle
        draw.ellipse([x, y, x + width, y + height], fill=200)

    # Add pleural effusion (fluid in lungs) - darker areas at bottom
    effusion_height = random.randint(20, 40)
    draw.rectangle([0, size[1] - effusion_height, size[0], size[1]], fill=140)

    # Add some linear opacities (Kerley B lines)
    for _ in range(random.randint(5, 10)):
        x1 = random.randint(100, size[0] - 100)
        y1 = random.randint(200, 400)
        x2 = x1 + random.randint(-20, 20)
        y2 = y1 + random.randint(10, 30)
        draw.line([(x1, y1), (x2, y2)], fill=160, width=2)

    # Clean up temp file
    if os.path.exists("temp_normal.jpg"):
        os.remove("temp_normal.jpg")

    img.save(filename)
    return img

def create_severe_pneumonia_xray(filename, size=(512, 512)):
    """Create a synthetic severe pneumonia X-ray"""
    # Start with normal X-ray
    img = create_normal_xray("temp_normal.jpg", size)
    draw = ImageDraw.Draw(img)

    # Add extensive consolidation (large white areas)
    consolidation_areas = random.randint(2, 4)

    for _ in range(consolidation_areas):
        x = random.randint(50, size[0] - 100)
        y = random.randint(100, 350)
        width = random.randint(80, 150)
        height = random.randint(60, 120)

        # Create large irregular consolidation
        draw.ellipse([x, y, x + width, y + height], fill=220)

    # Add bilateral pneumonia (both lungs affected)
    # Left lung consolidation
    draw.ellipse([50, 150, size[0]//2 - 10, 350], outline=200, width=5)
    # Right lung consolidation
    draw.ellipse([size[0]//2 + 10, 150, size[0] - 50, 350], outline=200, width=5)

    # Add pleural effusion (fluid)
    effusion_height = random.randint(30, 60)
    draw.rectangle([0, size[1] - effusion_height, size[0], size[1]], fill=120)

    # Add air bronchograms (dark lines in white areas)
    for _ in range(random.randint(8, 15)):
        x1 = random.randint(100, size[0] - 100)
        y1 = random.randint(150, 400)
        x2 = x1 + random.randint(-30, 30)
        y2 = y1 + random.randint(20, 50)
        draw.line([(x1, y1), (x2, y2)], fill=80, width=3)

    # Clean up temp file
    if os.path.exists("temp_normal.jpg"):
        os.remove("temp_normal.jpg")

    img.save(filename)
    return img

def create_test_dataset():
    """Create a comprehensive test dataset"""
    print("🏥 Creating X-Ray Test Dataset")
    print("=" * 50)

    # Create test_images directory
    os.makedirs("test_images", exist_ok=True)

    # Create different types of test images
    test_cases = [
        ("normal_1.jpg", "Normal Chest X-ray", create_normal_xray),
        ("normal_2.jpg", "Normal Chest X-ray (Variant)", create_normal_xray),
        ("pneumonia_mild_1.jpg", "Mild Pneumonia", create_pneumonia_xray),
        ("pneumonia_mild_2.jpg", "Mild Pneumonia (Variant)", create_pneumonia_xray),
        ("pneumonia_severe_1.jpg", "Severe Pneumonia", create_severe_pneumonia_xray),
        ("pneumonia_severe_2.jpg", "Severe Pneumonia (Bilateral)", create_severe_pneumonia_xray),
    ]

    print("🔄 Generating test images...")

    for filename, description, generator in test_cases:
        filepath = os.path.join("test_images", filename)
        print(f"   Creating: {description}")
        generator(filepath)
        print(f"   ✅ Saved: {filename}")

    print("\n📊 Test Dataset Summary:")
    print("=" * 50)

    # List all test images
    test_files = [f for f in os.listdir("test_images") if f.endswith('.jpg')]
    for i, filename in enumerate(test_files, 1):
        filepath = os.path.join("test_images", filename)
        size = os.path.getsize(filepath)
        print(f"   {i:2d}. {filename:<25} ({size:>6} bytes)")

    print(f"\n🎯 Total test images: {len(test_files)}")
    print("📁 Location: test_images/")
    print("\n💡 You can now use these images to test your Streamlit app!")

def main():
    create_test_dataset()

if __name__ == "__main__":
    main()
