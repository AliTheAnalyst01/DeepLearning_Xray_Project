# 🏥 X-Ray Pneumonia Classification API

A production-ready deep learning API for classifying chest X-ray images as Normal or Pneumonia using PyTorch and FastAPI.

## 🚀 Features

- **High Accuracy**: 96.67% accuracy with XRayCNN model
- **FastAPI Backend**: High-performance async API with automatic documentation
- **Streamlit Frontend**: Interactive web interface for image upload and prediction
- **Docker Support**: Containerized deployment with Docker Compose
- **Cloud Ready**: Deployed on Google Cloud Run
- **RESTful API**: Clean REST endpoints with JSON responses
- **Health Monitoring**: Built-in health checks and monitoring

## 📊 Model Performance

- **Architecture**: XRayCNN (Custom CNN)
- **Accuracy**: 96.67%
- **Classes**: Normal, Pneumonia
- **Input**: 224x224 RGB images
- **Framework**: PyTorch

## 🏗️ Project Structure

```
├── src/
│   └── app/
│       ├── api/           # FastAPI application
│       ├── models/        # ML model definitions
│       ├── services/      # Business logic
│       └── utils/         # Utility functions
├── Xray/                  # Core ML pipeline
├── data/
│   ├── raw/              # Test images
│   └── models/           # Trained models
├── deployment/           # Deployment configurations
├── scripts/              # Utility scripts
├── tests/                # Test suites
├── docs/                 # Documentation
├── config/               # Configuration files
└── logs/                 # Application logs
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Docker (optional)
- Google Cloud CLI (for cloud deployment)
- **Model File**: The trained model (`model.pt`) is not included in git due to size limits (105MB > 100MB GitHub limit)

### Local Development

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd DeepLearning_Xray_Project
   ```

2. **Create virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the API**

   ```bash
   python fast_api.py
   ```

5. **Run the Streamlit app**
   ```bash
   streamlit run streamlit_app.py
   ```

### Docker Deployment

1. **Build and run with Docker Compose**

   ```bash
   docker compose up -d
   ```

2. **Access the services**
   - API: http://localhost:8000
   - Streamlit: http://localhost:8501
   - API Docs: http://localhost:8000/docs

## 🌐 API Endpoints

### Health Check

```http
GET /health
```

**Response:**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_accuracy": "96.67%",
  "model_architecture": "XRayCNN",
  "device": "cuda"
}
```

### Prediction

```http
POST /predict
Content-Type: multipart/form-data
```

**Request:** Upload image file as `file`

**Response:**

```json
{
  "prediction": "NORMAL",
  "confidence": 0.9981,
  "class_probabilities": {
    "NORMAL": 0.9981,
    "PNEUMONIA": 0.0019
  },
  "model_info": {
    "architecture": "XRayCNN",
    "accuracy": "96.67%",
    "device": "cuda",
    "is_fallback": false
  },
  "status": "success"
}
```

## 🐳 Docker Commands

```bash
# Build image
docker build -t xray-api .

# Run container
docker run -p 8000:8000 xray-api

# Docker Compose
docker compose up -d          # Start services
docker compose down           # Stop services
docker compose logs -f        # View logs
docker compose restart        # Restart services
```

## ☁️ Cloud Deployment

### Google Cloud Run

1. **Install Google Cloud CLI**

   ```bash
   curl https://sdk.cloud.google.com | bash
   exec -l $SHELL
   ```

2. **Deploy to Cloud Run**
   ```bash
   gcloud auth login
   gcloud config set project your-project-id
   gcloud builds submit --tag gcr.io/your-project/xray-api
   gcloud run deploy xray-classification-api \
     --image gcr.io/your-project/xray-api \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --port 8000 \
     --memory 2Gi \
     --cpu 2
   ```

## 🧪 Testing

### Run Tests

```bash
# Unit tests
python -m pytest tests/unit/

# Integration tests
python -m pytest tests/integration/

# All tests
python -m pytest tests/
```

### Test API Endpoints

```bash
# Health check
curl -X GET "http://localhost:8000/health" | jq .

# Test prediction
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_images/normal_1.jpg" | jq .
```

## 📈 Monitoring

### Health Checks

- **API Health**: `GET /health`
- **Docker Health**: Built-in Docker health checks
- **Cloud Monitoring**: Google Cloud Run metrics

### Logs

```bash
# Docker logs
docker logs <container-name>

# Cloud Run logs
gcloud run logs tail --service=xray-classification-api --region=us-central1
```

## 🔧 Configuration

### Environment Variables

- `PYTHONUNBUFFERED=1` - Real-time logging
- `PYTHONPATH=/app` - Python path configuration

### Resource Limits

- **Memory**: 2GB (configurable)
- **CPU**: 2 cores (configurable)
- **Timeout**: 300 seconds
- **Concurrency**: 80 requests

## 📚 Documentation

- **API Documentation**: Available at `/docs` endpoint
- **Model Architecture**: XRayCNN with 96.67% accuracy
- **Deployment Guides**: See `docs/` directory
- **API Reference**: Interactive Swagger UI

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Medical Disclaimer

This tool is for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis. Always consult with qualified healthcare professionals for medical decisions.

## 🆘 Support

- **Issues**: Report bugs and feature requests
- **Documentation**: Check the docs/ directory
- **API Docs**: Visit `/docs` endpoint when running

## 🎯 Roadmap

- [ ] Model versioning
- [ ] Batch prediction API
- [ ] Model retraining pipeline
- [ ] Advanced monitoring dashboard
- [ ] Multi-cloud deployment support

---

**Built with ❤️ using FastAPI, PyTorch, and Streamlit**
