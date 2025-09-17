#!/bin/bash

# 🚀 X-Ray API Setup Script
# This script sets up the X-Ray Classification API project

set -e

echo "🚀 X-Ray API Setup"
echo "=================="

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/models
mkdir -p data/raw
mkdir -p logs

# Check if model file exists
if [ ! -f "data/models/model.pt" ]; then
    echo "📥 Model file not found. Please download it manually:"
    echo ""
    echo "Option 1: Download from Google Drive (if available)"
    echo "Option 2: Train the model using the training pipeline"
    echo "Option 3: Use a pre-trained model from the artifacts directory"
    echo ""
    echo "If you have the model file, place it at: data/models/model.pt"
    echo ""
    echo "For now, the API will use a fallback model for testing."
else
    echo "✅ Model file found: data/models/model.pt"
fi

# Check if test images exist
if [ ! -d "data/raw/test_images" ]; then
    echo "📥 Test images not found. Creating sample directory..."
    mkdir -p data/raw/test_images
    echo "Please add your test X-ray images to: data/raw/test_images/"
else
    echo "✅ Test images found: data/raw/test_images/"
fi

# Install dependencies
echo "📦 Installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✅ Dependencies installed"
else
    echo "❌ requirements.txt not found"
    exit 1
fi

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Warning: Virtual environment not activated"
    echo "   Consider running: python -m venv .venv && source .venv/bin/activate"
fi

echo ""
echo "✅ Setup completed!"
echo ""
echo "🚀 To start the API:"
echo "   python fast_api.py"
echo ""
echo "🎨 To start the Streamlit app:"
echo "   streamlit run streamlit_app.py"
echo ""
echo "🐳 To start with Docker:"
echo "   docker compose up -d"
echo ""
echo "📚 For more information, see README.md"
