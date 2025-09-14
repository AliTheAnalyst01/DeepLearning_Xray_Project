# 🚀 POSTMAN CONFIGURATION FOR X-RAY CLASSIFICATION API

## 📋 **API ENDPOINT DETAILS**

### 🌐 **Base URL:**
```
http://localhost:3000
```

### 🔧 **Prediction Endpoint:**
```
POST http://localhost:3000/predict
```

---

## 📱 **POSTMAN SETUP INSTRUCTIONS**

### **Step 1: Create New Request**
1. Open Postman
2. Click "New" → "Request"
3. Name it: "X-Ray Classification API"

### **Step 2: Configure Request**
1. **Method:** `POST`
2. **URL:** `http://localhost:3000/predict`

### **Step 3: Set Headers**
Go to **Headers** tab and add:
```
Key: Content-Type
Value: image/jpeg
```

### **Step 4: Set Body**
Go to **Body** tab:
1. Select **"binary"**
2. Click **"Select File"**
3. Choose any X-ray image (JPEG/PNG)

### **Step 5: Send Request**
Click **"Send"** button

---

## 📊 **EXPECTED RESPONSE**

### **Success Response (200 OK):**
```json
{
  "prediction": "NORMAL",
  "confidence": 0.9723,
  "class_probabilities": {
    "NORMAL": 0.9723,
    "PNEUMONIA": 0.0277
  }
}
```

### **Error Response (if any):**
```json
{
  "error": "Error message here"
}
```

---

## 🔧 **ALTERNATIVE TESTING METHODS**

### **Method 1: cURL Command**
```bash
curl -X POST "http://localhost:3000/predict" \
     -H "Content-Type: image/jpeg" \
     --data-binary @your_xray_image.jpg
```

### **Method 2: Python Script**
```python
import requests

url = "http://localhost:3000/predict"
headers = {"Content-Type": "image/jpeg"}

with open("your_xray_image.jpg", "rb") as f:
    response = requests.post(url, data=f, headers=headers)

print(response.json())
```

### **Method 3: JavaScript (Browser)**
```javascript
const formData = new FormData();
formData.append('image', fileInput.files[0]);

fetch('http://localhost:3000/predict', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

---

## 🚀 **STARTING THE SERVICE**

### **Command to Start Service:**
```bash
# Navigate to project directory
cd /home/faizan/Development/DeepLearning_Xray_Project

# Activate virtual environment
source .venv/bin/activate

# Navigate to deployment directory
cd artifacts/model_deployment

# Start the service
bentoml serve service:svc --port 3000
```

### **Verify Service is Running:**
```bash
# Check if service is running
curl http://localhost:3000/healthz

# Or check processes
ps aux | grep bentoml
```

---

## 📁 **TEST IMAGES**

### **Use Your Own X-Ray Images:**
- Any X-ray image in JPEG or PNG format
- Recommended size: 224x224 pixels (will be resized automatically)
- Supported formats: .jpg, .jpeg, .png

### **Sample Images from Your Dataset:**
- Location: `artifacts/20250914-143644/data_ingestion/data/test/`
- NORMAL images: `NORMAL/` folder
- PNEUMONIA images: `PNEUMONIA/` folder

---

## 🔍 **TROUBLESHOOTING**

### **If Service Won't Start:**
1. Make sure virtual environment is activated
2. Check if port 3000 is available
3. Try a different port: `--port 3001`

### **If Request Times Out:**
1. Wait a bit longer (model loading takes time)
2. Check service logs
3. Try with a smaller image

### **If You Get Connection Error:**
1. Verify service is running: `ps aux | grep bentoml`
2. Check if port is correct
3. Try restarting the service

---

## 📊 **API SPECIFICATION**

### **Request:**
- **Method:** POST
- **Content-Type:** image/jpeg or image/png
- **Body:** Binary image data

### **Response:**
- **Content-Type:** application/json
- **Status:** 200 OK (success) or 4xx/5xx (error)

### **Response Fields:**
- `prediction`: "NORMAL" or "PNEUMONIA"
- `confidence`: Float between 0 and 1
- `class_probabilities`: Object with probabilities for each class

---

## 🎯 **QUICK TEST CHECKLIST**

- [ ] Service is running (`ps aux | grep bentoml`)
- [ ] Port 3000 is accessible
- [ ] Postman request is configured correctly
- [ ] Image file is selected in Body tab
- [ ] Content-Type header is set to image/jpeg
- [ ] Request is sent successfully
- [ ] Response contains prediction and confidence

---

## 🎉 **SUCCESS INDICATORS**

✅ **Service Running:** Multiple bentoml processes visible  
✅ **Port Accessible:** `curl http://localhost:3000/healthz` returns response  
✅ **API Response:** JSON with prediction, confidence, and probabilities  
✅ **Model Working:** Predictions are reasonable (NORMAL/PNEUMONIA)  

---

**Your X-ray classification API is ready for real-world testing! 🚀**
