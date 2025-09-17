#!/bin/bash

# 🧹 Git Repository Cleanup and Preparation Script
# This script will clean up the repository and prepare it for a professional git push

set -e

echo "🧹 Git Repository Cleanup"
echo "========================="

# Remove unnecessary files and directories
echo "🗑️  Removing unnecessary files..."

# Remove large directories
rm -rf aws/ 2>/dev/null || true
rm -rf sam-installatio/ 2>/dev/null || true
rm -rf node_modules/ 2>/dev/null || true
rm -rf logs/ 2>/dev/null || true
rm -rf test/ 2>/dev/null || true
rm -rf tests/ 2>/dev/null || true

# Remove unnecessary files
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
rm -f test_*.py 2>/dev/null || true
rm -f cleanup_plan.md 2>/dev/null || true

# Remove old deployment guides (keep only README.md)
rm -f DEPLOYMENT_GUIDE.md 2>/dev/null || true
rm -f MODEL_DEPLOYMENT_GUIDE.md 2>/dev/null || true
rm -f FREE_DEPLOYMENT_OPTIONS.md 2>/dev/null || true
rm -f POSTMAN_CONFIGURATION.md 2>/dev/null || true
rm -f RAILWAY_DEPLOYMENT_GUIDE.md 2>/dev/null || true
rm -f DOCKER_DEPLOYMENT_GUIDE.md 2>/dev/null || true
rm -f GOOGLE_CLOUD_DEPLOYMENT.md 2>/dev/null || true
rm -f DEPLOYMENT_SUMMARY.md 2>/dev/null || true

# Remove old requirements files (keep only main one)
rm -f requirements_dev.txt 2>/dev/null || true
rm -f requirements_lambda.txt 2>/dev/null || true
rm -f requirements_streamlit.txt 2>/dev/null || true

# Remove old deployment scripts (keep only essential ones)
rm -f deploy-gcp-cloudrun.sh 2>/dev/null || true
rm -f quick-deploy-gcp.sh 2>/dev/null || true
rm -f deploy-docker-to-gcp.sh 2>/dev/null || true

# Clean up artifacts (keep only latest)
if [ -d "artifacts" ]; then
    echo "🧹 Cleaning artifacts directory..."
    # Keep only the latest artifact directory
    LATEST_ARTIFACT=$(ls -t artifacts/ 2>/dev/null | head -1)
    if [ -n "$LATEST_ARTIFACT" ]; then
        echo "Keeping latest artifact: $LATEST_ARTIFACT"
        # Create a temporary directory for the latest artifact
        mkdir -p temp_artifacts
        mv "artifacts/$LATEST_ARTIFACT" temp_artifacts/
        rm -rf artifacts/
        mv temp_artifacts/$LATEST_ARTIFACT artifacts/
        rmdir temp_artifacts
    fi
fi

# Create essential directories
echo "📁 Creating essential directories..."
mkdir -p docs scripts tests

# Move essential files to proper locations
echo "📦 Organizing files..."

# Move deployment files
mkdir -p deployment
mv Dockerfile docker-compose.yml docker-compose.prod.yml cloudbuild.yaml deployment/ 2>/dev/null || true

# Move scripts
mv deploy.sh scripts/ 2>/dev/null || true

# Move test images to data directory
mkdir -p data/raw
mv test_images/ data/raw/ 2>/dev/null || true

# Move model to data directory
mkdir -p data/models
mv model.pt data/models/ 2>/dev/null || true

# Initialize git if not already initialized
if [ ! -d ".git" ]; then
    echo "🔧 Initializing git repository..."
    git init
fi

# Add all files to git
echo "📝 Adding files to git..."
git add .

# Check git status
echo "📊 Git status:"
git status

echo "✅ Cleanup completed!"
echo "🎯 Repository is ready for professional git push"
echo ""
echo "Next steps:"
echo "1. Review the changes: git status"
echo "2. Commit the changes: git commit -m 'Clean up project structure'"
echo "3. Add remote origin: git remote add origin <your-repo-url>"
echo "4. Push to repository: git push -u origin main"
