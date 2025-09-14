# 🚀 Model Deployment Guide

This guide explains how to deploy your trained X-ray classification model using BentoML.

## 📋 Prerequisites

1. **BentoML Installation**

   ```bash
   pip install bentoml
   ```

2. **Required Dependencies**
   ```bash
   pip install torch torchvision pillow numpy
   ```

## 🔧 Model Pusher Implementation

### **Step 1: Understanding the ModelPusher Class**

The `ModelPusher` class handles the complete deployment process:

```python
class ModelPusher:
    def __init__(self, model_pusher_config: ModelPusherConfig):
        # Initialize with BentoML configuration
        self.model_pusher_config = model_pusher_config
        self.deployment_dir = os.path.join("artifacts", "model_deployment")
```

### **Step 2: Key Methods Explained**

#### **1. `_load_trained_model()` - Model Loading**

```python
def _load_trained_model(self, model_path: str) -> tuple:
    # Load PyTorch model state dict
    model_state = torch.load(model_path, map_location='cpu')

    # Handle different save formats
    if isinstance(model_state, dict) and 'model_state_dict' in model_state:
        # Custom format with metadata
        model = XRayCNN(num_classes=model_state.get('num_classes', 2))
        model.load_state_dict(model_state['model_state_dict'])
        metadata = {
            'accuracy': model_state.get('accuracy', 0.0),
            'loss': model_state.get('loss', 0.0),
            'epochs': model_state.get('epochs', 0),
            'num_classes': model_state.get('num_classes', 2)
        }
    else:
        # Direct state dict
        model = XRayCNN(num_classes=2)
        model.load_state_dict(model_state)
        metadata = {'num_classes': 2}

    model.eval()  # Set to evaluation mode
    return model, metadata
```

**What this does:**

- Loads the trained PyTorch model
- Extracts metadata (accuracy, loss, etc.)
- Handles different model save formats
- Sets model to evaluation mode

#### **2. `_create_bentoml_model()` - BentoML Packaging**

```python
def _create_bentoml_model(self, model, metadata: dict, model_trainer_artifact) -> str:
    # Generate version with timestamp and accuracy
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    accuracy = model_trainer_artifact.test_accuracy
    version = f"v{timestamp}_acc{accuracy:.3f}"

    # Define save function for BentoML
    def model_save_func(model_obj):
        torch.save(model_obj.state_dict(), "model_state_dict.pt")
        model_info = {
            'num_classes': metadata.get('num_classes', 2),
            'model_type': 'XRayCNN',
            'input_size': (3, 224, 224),
            'class_names': ['NORMAL', 'PNEUMONIA']
        }
        with open("model_info.json", "w") as f:
            json.dump(model_info, f)

    # Define load function for BentoML
    def model_load_func(model_path):
        with open(os.path.join(model_path, "model_info.json"), "r") as f:
            model_info = json.load(f)

        model_obj = XRayCNN(num_classes=model_info['num_classes'])
        state_dict = torch.load(os.path.join(model_path, "model_state_dict.pt"), map_location='cpu')
        model_obj.load_state_dict(state_dict)
        model_obj.eval()
        return model_obj

    # Save with BentoML
    bentoml_model = save_model(
        name=self.model_pusher_config.bentoml_model_name,
        model=model,
        model_save_func=model_save_func,
        model_load_func=model_load_func,
        metadata={
            'accuracy': accuracy,
            'loss': model_trainer_artifact.test_loss,
            'timestamp': timestamp,
            'version': version,
            'framework': 'pytorch',
            'task': 'image_classification'
        },
        version=version
    )

    return str(bentoml_model.tag)
```

**What this does:**

- Creates versioned model packages
- Defines how to save/load the model
- Adds metadata for tracking
- Uses BentoML's model registry

#### **3. `_create_prediction_service()` - API Service**

```python
def _create_prediction_service(self, model_tag: str, transforms_path: str) -> str:
    service_code = f'''
import bentoml
import torch
import torchvision.transforms as transforms
from PIL import Image
from bentoml.io import Image as BentoImage, JSON

# Load the model
model_runner = bentoml.pytorch.get("{model_tag}").to_runner()

# Define preprocessing pipeline
transforms_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create the service
svc = bentoml.Service("{self.model_pusher_config.bentoml_service_name}", runners=[model_runner])

@svc.api(input=BentoImage(), output=JSON())
def predict(input_image: Image.Image) -> dict:
    # Preprocess the image
    input_tensor = transforms_pipeline(input_image).unsqueeze(0)

    # Get prediction
    with torch.no_grad():
        outputs = model_runner.run(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    # Return results
    class_names = ['NORMAL', 'PNEUMONIA']
    predicted_class_name = class_names[predicted.item()]

    return {{
        "prediction": predicted_class_name,
        "confidence": round(confidence.item(), 4),
        "class_probabilities": {{
            "NORMAL": round(probabilities[0][0].item(), 4),
            "PNEUMONIA": round(probabilities[0][1].item(), 4)
        }}
    }}
'''

    # Save service file
    service_file_path = os.path.join(self.deployment_dir, "service.py")
    with open(service_file_path, 'w') as f:
        f.write(service_code)

    return service_file_path
```

**What this does:**

- Creates a REST API service
- Handles image preprocessing
- Returns JSON predictions
- Uses BentoML's API decorators

#### **4. `_create_bentofile()` - Deployment Configuration**

```python
def _create_bentofile(self, service_path: str) -> str:
    bentofile_content = f'''service: "{self.model_pusher_config.bentoml_service_name}"
include:
  - "{service_path}"
python:
  packages:
    - torch
    - torchvision
    - pillow
    - numpy
    - bentoml
  requirements_txt: requirements.txt

docker:
  distro: debian
  python_version: "3.9"
  cuda_version: "11.8"
  env:
    - BENTOML_HOME=/home/bentoml
'''

    bentofile_path = os.path.join(self.deployment_dir, "bentofile.yaml")
    with open(bentofile_path, 'w') as f:
        f.write(bentofile_content)

    return bentofile_path
```

**What this does:**

- Defines service configuration
- Specifies Python dependencies
- Configures Docker settings
- Sets up environment variables

## 🚀 Usage Examples

### **1. Run Complete Pipeline with Model Pusher**

```python
from Xray.pipeline.training_pipeline import TrainPipeline

# Initialize pipeline
train_pipeline = TrainPipeline()

# Run complete pipeline (includes model pusher)
train_pipeline.run_pipeline()
```

### **2. Test Model Pusher Only**

```python
python test_model_pusher.py
```

### **3. Deploy the Model**

```bash
# Navigate to deployment directory
cd artifacts/model_deployment

# Build BentoML service
bentoml build

# Serve locally
bentoml serve xray_service:latest

# Test the API
curl -X POST "http://localhost:3000/predict" \
     -H "Content-Type: image/jpeg" \
     --data-binary @test_image.jpg
```

## 📁 Generated Artifacts

After running the model pusher, you'll find these files in `artifacts/model_deployment/`:

```
artifacts/model_deployment/
├── service.py              # BentoML prediction service
├── bentofile.yaml          # Service configuration
├── requirements.txt        # Python dependencies
└── deployment_info.json    # Deployment metadata
```

## 🔍 Model Versioning

The model pusher automatically creates versions based on:

- **Timestamp**: `YYYYMMDD_HHMMSS`
- **Accuracy**: Model performance
- **Format**: `v{timestamp}_acc{accuracy:.3f}`

Example: `v20250113_143022_acc0.967`

## 🌐 Deployment Options

### **1. Local Deployment**

```bash
bentoml serve xray_service:latest
```

### **2. Docker Deployment**

```bash
bentoml containerize xray_service:latest
docker run -p 3000:3000 xray_service:latest
```

### **3. Cloud Deployment**

```bash
# Deploy to AWS ECS
bentoml ecs deploy xray_service:latest

# Deploy to Google Cloud Run
bentoml gcp deploy xray_service:latest
```

## 📊 Monitoring and Logging

The model pusher includes comprehensive logging:

- Model loading status
- BentoML packaging progress
- Service creation steps
- Deployment artifact paths

Check logs in the `logs/` directory for detailed information.

## 🛠️ Troubleshooting

### **Common Issues:**

1. **BentoML not installed**

   ```bash
   pip install bentoml
   ```

2. **Model loading errors**

   - Check model file path
   - Verify model architecture matches

3. **Service creation fails**

   - Check transforms file path
   - Verify BentoML model exists

4. **Deployment issues**
   - Check bentofile.yaml syntax
   - Verify all dependencies are listed

## 📈 Next Steps

1. **Test the deployed model** with sample images
2. **Monitor performance** and accuracy
3. **Set up CI/CD** for automated deployment
4. **Add monitoring** and alerting
5. **Scale the service** based on demand

---

**Happy Deploying! 🚀**
