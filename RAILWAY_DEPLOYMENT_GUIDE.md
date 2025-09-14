# 🚀 RAILWAY DEPLOYMENT GUIDE

## 📋 **STEP-BY-STEP DEPLOYMENT**

### **Step 1: Install Railway CLI**
```bash
# Install Node.js first (if not installed)
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Install Railway CLI
npm install -g @railway/cli
```

### **Step 2: Login to Railway**
```bash
railway login
```
- This will open your browser to authenticate with Railway

### **Step 3: Initialize Railway Project**
```bash
railway init
```
- Choose "Empty Project"
- This creates a `railway.json` configuration

### **Step 4: Deploy to Railway**
```bash
railway up
```
- This builds and deploys your Docker container
- Railway will give you a public URL

### **Step 5: Test Your Deployed API**
- Your API will be available at: `https://your-app-name.railway.app`
- Health check: `https://your-app-name.railway.app/health`
- API docs: `https://your-app-name.railway.app/docs`

---

## 🔧 **ALTERNATIVE: GitHub Integration**

### **Method 1: Connect GitHub Repository**
1. Go to [railway.app](https://railway.app)
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Choose your repository
5. Railway will auto-deploy on every push

### **Method 2: Manual Upload**
1. Go to [railway.app](https://railway.app)
2. Click "New Project"
3. Select "Deploy from template"
4. Choose "Dockerfile"
5. Upload your project files

---

## 📱 **POSTMAN TESTING AFTER DEPLOYMENT**

### **Your Deployed API Endpoints:**
- **Base URL:** `https://your-app-name.railway.app`
- **Health Check:** `GET https://your-app-name.railway.app/health`
- **Prediction:** `POST https://your-app-name.railway.app/predict`
- **API Docs:** `GET https://your-app-name.railway.app/docs`

### **Postman Configuration:**
1. **Method:** POST
2. **URL:** `https://your-app-name.railway.app/predict`
3. **Body:** form-data
4. **Key:** file (Type: File)
5. **Value:** Select any X-ray image

---

## 🎯 **EXPECTED RESPONSE**
```json
{
  "prediction": "NORMAL",
  "confidence": 0.5027,
  "class_probabilities": {
    "NORMAL": 0.5027,
    "PNEUMONIA": 0.4973
  },
  "status": "success"
}
```

---

## 🔍 **TROUBLESHOOTING**

### **If Deployment Fails:**
1. Check Railway logs: `railway logs`
2. Verify Dockerfile syntax
3. Ensure all dependencies are in requirements.txt

### **If API Doesn't Respond:**
1. Check health endpoint first
2. Verify port 8000 is exposed
3. Check Railway service status

### **If Model Loading Fails:**
1. Ensure model files are included
2. Check file paths in the code
3. Verify Python dependencies

---

## 🎉 **SUCCESS INDICATORS**

✅ **Deployment Successful:** Railway shows "Deployed" status  
✅ **Health Check Passes:** `/health` returns 200 OK  
✅ **API Responds:** `/predict` accepts image uploads  
✅ **Predictions Work:** Returns valid JSON responses  

---

## 📊 **RAILWAY FREE TIER LIMITS**

- **Build Time:** 500 minutes/month
- **Deploy Time:** 100 hours/month
- **Bandwidth:** 100GB/month
- **Storage:** 1GB

**Perfect for your X-ray classification API!** 🚀
