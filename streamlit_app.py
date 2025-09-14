#!/usr/bin/env python3
"""
Streamlit Web App for X-Ray Pneumonia Classification
Uses the deployed FastAPI endpoint for predictions
"""

import streamlit as st
import requests
import json
from PIL import Image
import io
import numpy as np
import time

# Page configuration
st.set_page_config(
    page_title="🏥 X-Ray Pneumonia Classifier",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .confidence-bar {
        height: 30px;
        background: linear-gradient(90deg, #ff6b6b 0%, #4ecdc4 100%);
        border-radius: 15px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
    }
    .upload-section {
        border: 2px dashed #1f77b4;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "https://xray-classifier.onrender.com"  # Update with your deployed URL
LOCAL_API_URL = "http://localhost:8001"  # For local testing

def check_api_health(api_url):
    """Check if the API is healthy and running"""
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except:
        return False, None

def predict_xray(image, api_url):
    """Send image to API for prediction"""
    try:
        # Convert PIL image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        # Send request to API
        files = {'file': ('xray.jpg', img_byte_arr, 'image/jpeg')}
        response = requests.post(f"{api_url}/predict", files=files, timeout=30)

        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"API Error: {response.status_code} - {response.text}"
    except Exception as e:
        return False, f"Connection Error: {str(e)}"

def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 X-Ray Pneumonia Classifier</h1>', unsafe_allow_html=True)
    st.markdown("### Upload a chest X-ray image to detect pneumonia using AI")

    # Sidebar
    st.sidebar.title("🔧 Configuration")

    # API URL selection
    api_choice = st.sidebar.radio(
        "Select API Endpoint:",
        ["🏠 Local API", "🌐 Deployed API (Render)"],
        help="Choose between local development API or deployed cloud API"
    )

    api_url = LOCAL_API_URL if api_choice == "🏠 Local API" else API_BASE_URL

    # Check API health
    with st.sidebar:
        st.subheader("📊 API Status")
        with st.spinner("Checking API health..."):
            is_healthy, health_data = check_api_health(api_url)

        if is_healthy:
            st.success("✅ API is healthy and running")
            if health_data:
                st.json(health_data)
        else:
            st.error("❌ API is not responding")
            st.info("Make sure your API is running locally or deployed correctly")

    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.subheader("📤 Upload X-Ray Image")

        uploaded_file = st.file_uploader(
            "Choose an X-ray image file",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a chest X-ray image in JPG, JPEG, or PNG format"
        )

        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded X-Ray Image", use_column_width=True)

            # Image info
            st.info(f"📏 Image size: {image.size[0]} x {image.size[1]} pixels")
            st.info(f"📁 File size: {len(uploaded_file.getvalue())} bytes")

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.subheader("🔍 Prediction Results")

        if uploaded_file is not None:
            if st.button("🚀 Analyze X-Ray", type="primary", use_container_width=True):
                with st.spinner("🤖 AI is analyzing your X-ray image..."):
                    success, result = predict_xray(image, api_url)

                if success:
                    # Display prediction results
                    prediction = result['prediction']
                    confidence = result['confidence']
                    probabilities = result['class_probabilities']

                    # Prediction box
                    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                    st.markdown(f"## 🎯 Prediction: {prediction}")
                    st.markdown(f"### Confidence: {confidence:.1%}")
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Confidence visualization
                    st.subheader("📊 Confidence Breakdown")

                    # Normal probability
                    normal_prob = probabilities['NORMAL']
                    st.metric("Normal", f"{normal_prob:.1%}")
                    st.progress(normal_prob)

                    # Pneumonia probability
                    pneumonia_prob = probabilities['PNEUMONIA']
                    st.metric("Pneumonia", f"{pneumonia_prob:.1%}")
                    st.progress(pneumonia_prob)

                    # Interpretation
                    if prediction == "NORMAL":
                        st.success("✅ This X-ray appears to be normal. No signs of pneumonia detected.")
                    else:
                        st.warning("⚠️ This X-ray shows signs of pneumonia. Please consult a healthcare professional.")

                    # Raw API response
                    with st.expander("🔍 View Raw API Response"):
                        st.json(result)

                else:
                    st.error(f"❌ Prediction failed: {result}")
                    st.info("Please check your API connection and try again.")

        else:
            st.info("👆 Please upload an X-ray image to get started")

    # Footer
    st.markdown("---")
    st.markdown("### 📚 About This App")
    st.markdown("""
    This application uses a deep learning model to classify chest X-ray images as either **Normal** or **Pneumonia**.

    **How it works:**
    1. Upload a chest X-ray image
    2. The image is sent to our AI model via API
    3. The model analyzes the image and provides a prediction
    4. Results show the prediction and confidence scores

    **⚠️ Medical Disclaimer:**
    This tool is for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis. Always consult with qualified healthcare professionals for medical decisions.
    """)

    # API Information
    with st.expander("🔗 API Information"):
        st.markdown(f"**Current API URL:** `{api_url}`")
        st.markdown("**Available Endpoints:**")
        st.markdown(f"- Health Check: `{api_url}/health`")
        st.markdown(f"- Prediction: `{api_url}/predict`")
        st.markdown(f"- Documentation: `{api_url}/docs`")

if __name__ == "__main__":
    main()
