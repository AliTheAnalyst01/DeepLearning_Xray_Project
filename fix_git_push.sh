#!/bin/bash

# 🔧 Fix Git Push Issue - Large File Problem
# This script fixes the git push issue caused by large model files

set -e

echo "🔧 Fixing Git Push Issue"
echo "======================="

# Remove large files from git tracking
echo "🗑️  Removing large files from git tracking..."

# Remove model files from git
git rm --cached data/models/model.pt 2>/dev/null || true
git rm --cached model.pt 2>/dev/null || true

# Remove any other large files
find . -name "*.pt" -size +50M -exec git rm --cached {} \; 2>/dev/null || true

# Add updated .gitignore
echo "📝 Updating .gitignore..."
git add .gitignore

# Commit the changes
echo "💾 Committing changes..."
git add .
git commit -m "Remove large model files from git tracking

- Exclude .pt files from git due to GitHub 100MB limit
- Add setup.sh for model file download instructions
- Update .gitignore to exclude large model files
- Model files should be downloaded separately or trained locally"

echo "✅ Git repository fixed!"
echo ""
echo "📋 Next steps:"
echo "1. Push to repository: git push"
echo "2. Add model file locally: cp artifacts/20250914-182235/model_training/model.pt data/models/"
echo "3. Or train a new model using the training pipeline"
echo ""
echo "🎯 The API will work with a fallback model if model.pt is not available"
