#!/usr/bin/env python3
"""
Deploy X-Ray Classification API to Render
This script helps you prepare and deploy your API
"""

import os
import subprocess
import json

def check_git_status():
    """Check if code is committed to git"""
    try:
        result = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True)
        if result.stdout.strip():
            print("⚠️  You have uncommitted changes:")
            print(result.stdout)
            return False
        else:
            print("✅ Git repository is clean")
            return True
    except Exception as e:
        print(f"❌ Git check failed: {e}")
        return False

def create_render_config():
    """Create render.yaml configuration"""
    render_config = {
        "services": [
            {
                "type": "web",
                "name": "xray-classifier-api",
                "env": "python",
                "buildCommand": "pip install -r requirements.txt",
                "startCommand": "python3 fast_api.py",
                "envVars": [
                    {
                        "key": "PYTHON_VERSION",
                        "value": "3.12.0"
                    }
                ]
            }
        ]
    }

    with open('render.yaml', 'w') as f:
        import yaml
        yaml.dump(render_config, f, default_flow_style=False)

    print("✅ Created render.yaml configuration")

def update_requirements():
    """Update requirements.txt for deployment"""
    requirements = [
        "fastapi==0.104.1",
        "uvicorn[standard]==0.24.0",
        "python-multipart==0.0.6",
        "torch==2.1.0",
        "torchvision==0.16.0",
        "Pillow==10.0.0",
        "numpy==1.24.0",
        "scikit-learn==1.3.0",
        "matplotlib==3.7.2",
        "seaborn==0.12.2"
    ]

    with open('requirements.txt', 'w') as f:
        f.write('\n'.join(requirements))

    print("✅ Updated requirements.txt for deployment")

def create_startup_script():
    """Create startup script for Render"""
    startup_script = """#!/bin/bash
echo "🚀 Starting X-Ray Classification API on Render..."
echo "📊 Model: XRayCNN with 96.67% accuracy"
echo "🌐 Port: $PORT"
python3 fast_api.py
"""

    with open('start.sh', 'w') as f:
        f.write(startup_script)

    os.chmod('start.sh', 0o755)
    print("✅ Created start.sh script")

def main():
    print("🚀 X-Ray API Deployment Preparation")
    print("=" * 50)

    # Check git status
    if not check_git_status():
        print("\n💡 Please commit your changes first:")
        print("   git add .")
        print("   git commit -m 'Prepare for deployment'")
        print("   git push origin main")
        return

    # Create deployment files
    create_render_config()
    update_requirements()
    create_startup_script()

    print("\n📋 Deployment Checklist:")
    print("=" * 50)
    print("✅ 1. Code committed to GitHub")
    print("✅ 2. render.yaml created")
    print("✅ 3. requirements.txt updated")
    print("✅ 4. start.sh created")

    print("\n🌐 Next Steps:")
    print("=" * 50)
    print("1. Go to https://render.com")
    print("2. Sign up/Login with GitHub")
    print("3. Click 'New +' → 'Web Service'")
    print("4. Connect your GitHub repository")
    print("5. Use these settings:")
    print("   - Build Command: pip install -r requirements.txt")
    print("   - Start Command: python3 fast_api.py")
    print("   - Environment: Python 3")
    print("6. Deploy and get your URL!")

    print("\n🎯 Your API will be available at:")
    print("   https://your-app-name.onrender.com")
    print("   Health: /health")
    print("   Predict: /predict")
    print("   Docs: /docs")

    print("\n💡 After deployment, update your Streamlit app:")
    print("   API_BASE_URL = 'https://your-app-name.onrender.com'")

if __name__ == "__main__":
    main()
