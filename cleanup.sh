#!/bin/bash

# 🧹 Project Cleanup Script
# This script will clean up the project directory and create a professional structure

set -e

echo "🧹 Starting Project Cleanup..."
echo "================================"

# Create backup directory
echo "📦 Creating backup..."
mkdir -p .backup/$(date +%Y%m%d_%H%M%S)

# Remove large unnecessary directories
echo "🗑️  Removing large unnecessary directories..."
rm -rf aws/ 2>/dev/null || true
rm -rf sam-installatio/ 2>/dev/null || true
rm -rf node_modules/ 2>/dev/null || true
rm -rf logs/ 2>/dev/null || true
rm -rf test/ 2>/dev/null || true
rm -rf tests/ 2>/dev/null || true

# Remove unnecessary files
echo "🗑️  Removing unnecessary files..."
rm -f aws-sam-cli-linux-x86_64.zip 2>/dev/null || true
rm -f package.json package-lock.json 2>/dev/null || true
rm -f bentofile.yaml 2>/dev/null || true
rm -f serverless.yml serverless.yml.backup 2>/dev/null || true
rm -f railway.json 2>/dev/null || true
rm -f render.yaml 2>/dev/null || true
rm -f lambda_handler.py lambda_model.pt 2>/dev/null || true
rm -f check_models.py 2>/dev/null || true
rm -f create_*.py 2>/dev/null || true
rm -f download_*.py 2>/dev/null || true
rm -f view_and_test_model.py 2>/dev/null || true
rm -f verify-docker.sh 2>/dev/null || true
rm -f start.sh 2>/dev/null || true
rm -f MODEL_SUMMARY.py 2>/dev/null || true
rm -f main_transformation_only.py 2>/dev/null || true
rm -f main.py 2>/dev/null || true
rm -f setup.py 2>/dev/null || true
rm -f deploy_render.py 2>/dev/null || true

# Remove old artifacts (keep only latest)
echo "🗑️  Cleaning old artifacts..."
if [ -d "artifacts" ]; then
    # Keep only the latest artifact directory
    LATEST_ARTIFACT=$(ls -t artifacts/ | head -1)
    echo "Keeping latest artifact: $LATEST_ARTIFACT"
    # Move old artifacts to backup
    for dir in artifacts/*/; do
        if [ "$dir" != "artifacts/$LATEST_ARTIFACT/" ]; then
            echo "Moving $dir to backup"
            mv "$dir" .backup/$(date +%Y%m%d_%H%M%S)/ 2>/dev/null || true
        fi
    done
fi

# Remove old deployment guides (keep only essential ones)
echo "🗑️  Cleaning documentation..."
rm -f DEPLOYMENT_GUIDE.md 2>/dev/null || true
rm -f MODEL_DEPLOYMENT_GUIDE.md 2>/dev/null || true
rm -f FREE_DEPLOYMENT_OPTIONS.md 2>/dev/null || true
rm -f POSTMAN_CONFIGURATION.md 2>/dev/null || true
rm -f RAILWAY_DEPLOYMENT_GUIDE.md 2>/dev/null || true
rm -f DOCKER_DEPLOYMENT_GUIDE.md 2>/dev/null || true
rm -f GOOGLE_CLOUD_DEPLOYMENT.md 2>/dev/null || true
rm -f DEPLOYMENT_SUMMARY.md 2>/dev/null || true
rm -f cleanup_plan.md 2>/dev/null || true

# Remove old test files
echo "🗑️  Cleaning test files..."
rm -f test_*.py 2>/dev/null || true

# Remove old requirements files (keep only main one)
echo "🗑️  Cleaning requirements files..."
rm -f requirements_dev.txt 2>/dev/null || true
rm -f requirements_lambda.txt 2>/dev/null || true
rm -f requirements_streamlit.txt 2>/dev/null || true

# Remove old deployment scripts (keep only working ones)
echo "🗑️  Cleaning deployment scripts..."
rm -f deploy-gcp-cloudrun.sh 2>/dev/null || true
rm -f quick-deploy-gcp.sh 2>/dev/null || true

# Create clean directory structure
echo "📁 Creating clean directory structure..."
mkdir -p src/app/{api,models,services,utils}
mkdir -p docs
mkdir -p scripts/{deployment,testing}
mkdir -p tests/{unit,integration}
mkdir -p config
mkdir -p data/{raw,processed,models}
mkdir -p logs

# Move essential files to proper locations
echo "📦 Organizing files..."

# Move API files
mv fast_api.py src/app/api/
mv streamlit_app.py src/app/

# Move core ML code
mv Xray/ src/app/

# Move deployment files
mv Dockerfile docker-compose.yml docker-compose.prod.yml deployment/
mv deploy-docker-to-gcp.sh scripts/deployment/
mv cloudbuild.yaml deployment/

# Move test images
mv test_images/ data/raw/

# Move model files
mv model.pt data/models/
if [ -d "artifacts" ]; then
    mv artifacts/ data/models/
fi

# Move configuration files
mv requirements.txt config/

echo "✅ Cleanup completed!"
echo "📊 Project structure cleaned and organized"
echo "🎯 Ready for professional git push"
