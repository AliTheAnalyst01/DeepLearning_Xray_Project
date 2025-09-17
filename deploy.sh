#!/bin/bash

# 🚀 X-Ray API Deployment Script
# Simple deployment script for the X-Ray Classification API

set -e

echo "🚀 X-Ray API Deployment"
echo "======================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Build Docker image
echo "📦 Building Docker image..."
docker build -t xray-api .

# Stop existing containers
echo "🛑 Stopping existing containers..."
docker compose down 2>/dev/null || true

# Start services
echo "🚀 Starting services..."
docker compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 10

# Test the API
echo "🧪 Testing API..."
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ API is running successfully!"
    echo "🌐 API URL: http://localhost:8000"
    echo "📊 Health Check: http://localhost:8000/health"
    echo "🔍 Prediction: http://localhost:8000/predict"
    echo "📚 API Docs: http://localhost:8000/docs"
    echo "🎨 Streamlit App: http://localhost:8501"
else
    echo "❌ API failed to start. Check logs with: docker compose logs"
    exit 1
fi

echo "🎉 Deployment completed successfully!"
