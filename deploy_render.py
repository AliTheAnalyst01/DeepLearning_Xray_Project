#!/usr/bin/env python3
"""
Deploy X-Ray Classification API to Render
"""

import subprocess
import os

def deploy_to_render():
    print("🚀 Deploying X-Ray Classification API to Render...")
    print("=" * 60)
    
    # Check if git is clean
    try:
        result = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True)
        if result.stdout.strip():
            print("⚠️  You have uncommitted changes. Committing them...")
            subprocess.run(['git', 'add', '.'], check=True)
            subprocess.run(['git', 'commit', '-m', 'Deploy to Render with 96.67% accuracy model'], check=True)
            subprocess.run(['git', 'push', 'origin', 'main'], check=True)
            print("✅ Changes committed and pushed")
        else:
            print("✅ Git repository is clean")
    except Exception as e:
        print(f"❌ Git error: {e}")
        return False
    
    print("\n📋 Deployment Instructions:")
    print("=" * 60)
    print("1. Go to https://render.com")
    print("2. Sign up/Login with GitHub")
    print("3. Click 'New +' → 'Web Service'")
    print("4. Connect repository: AliTheAnalyst01/DeepLearning_Xray_Project")
    print("5. Use these settings:")
    print("   - Build Command: pip install -r requirements.txt")
    print("   - Start Command: python3 fast_api.py")
    print("   - Environment: Python 3")
    print("   - Instance Type: Free")
    print("6. Click 'Create Web Service'")
    print("7. Wait for deployment (5-10 minutes)")
    
    print("\n🎯 After deployment:")
    print("=" * 60)
    print("Your API will be available at:")
    print("https://your-app-name.onrender.com")
    print("\nEndpoints:")
    print("- Health: /health")
    print("- Predict: /predict")
    print("- Docs: /docs")
    
    print("\n💡 Update your Streamlit app:")
    print("API_BASE_URL = 'https://your-app-name.onrender.com'")
    
    return True

if __name__ == "__main__":
    deploy_to_render()
