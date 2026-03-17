# 🧬 Medical Image AI System

> **Production-ready multi-disease detection from medical images with Grad-CAM explainability**

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange?logo=tensorflow)](https://tensorflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-ready-blue?logo=docker)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [System Architecture](#architecture)
3. [Supported Diseases](#diseases)
4. [Project Structure](#structure)
5. [Quick Start](#quickstart)
6. [Training Your Own Model](#training)
7. [API Reference](#api)
8. [Frontend](#frontend)
9. [Docker Deployment](#docker)
10. [Cloud Deployment](#cloud)
11. [GPU Support](#gpu)
12. [Model Monitoring](#monitoring)

---

## 🌟 Overview <a name="overview"></a>

Medical AI System is an end-to-end deep learning platform for automated disease detection from medical images. It exposes a FastAPI REST API, produces Grad-CAM visual explanations, and ships with a real-time monitoring dashboard.

**Key capabilities:**
- 🧠 Brain tumor detection from MRI scans (4 classes)
- 🫁 Pneumonia detection from chest X-rays (2 classes)
- 🔬 Skin cancer classification (7 classes – HAM10000)
- 🔥 Grad-CAM heatmaps showing **why** the model made its decision
- ⚡ Async FastAPI with concurrent request handling
- 📊 Real-time monitoring dashboard
- 🐳 Docker-containerised and cloud-deployable

---

## 🏗️ System Architecture <a name="architecture"></a>

```
┌─────────────────────────────────────────────────────────────┐
│                     React Frontend                          │
│   Upload Image → Select Disease → View Result + Heatmap    │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP (multipart/form-data)
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   FastAPI Backend                           │
│                                                             │
│  POST /predict                                              │
│    ├─ preprocess.py   → resize, normalise                  │
│    ├─ model_loader.py → load Keras model                   │
│    ├─ inference.py    → async predict                      │
│    └─ gradcam.py      → Grad-CAM overlay                   │
│                                                             │
│  GET  /monitoring     → stats dashboard                     │
│  GET  /health         → liveness probe                     │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│              MobileNetV2 Transfer Learning                  │
│   ImageNet backbone → GlobalAvgPool → Dropout → Dense      │
│   Saved as model/<disease>_model.keras                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 🩺 Supported Diseases <a name="diseases"></a>

| Disease | Modality | Classes | Dataset |
|---|---|---|---|
| Brain Tumor | MRI | glioma, meningioma, no_tumor, pituitary | [Kaggle Brain MRI](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) |
| Pneumonia | Chest X-Ray | normal, pneumonia | [Chest X-Ray Images](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) |
| Skin Cancer | Dermoscopy | melanoma, nevi, BCC, AK, DF, VL, BKL | [HAM10000](https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection) |

---

## 📁 Project Structure <a name="structure"></a>

```
medical-ai-system/
│
├── app/                        # FastAPI application
│   ├── main.py                 # Routes, CORS, monitoring store
│   ├── inference.py            # Async inference orchestrator
│   ├── preprocess.py           # Image decode → tensor pipeline
│   ├── model_loader.py         # Model registry, load_all_models()
│   └── gradcam.py              # Grad-CAM + heatmap overlay
│
├── model/                      # Trained Keras models
│   ├── brain_tumor_model.keras
│   ├── pneumonia_model.keras
│   └── skin_cancer_model.keras
│
├── training/
│   └── train_model.py          # Full 2-phase training pipeline
│
├── frontend/
│   └── index.html              # Single-file React app (no build needed)
│
├── docker/
│   ├── Dockerfile              # Multi-stage production build
│   └── docker-compose.yml      # API + frontend stack
│
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start <a name="quickstart"></a>

### Prerequisites
- Python 3.11+
- Node.js 18+ (for frontend dev server, optional)
- Docker (for containerised deployment)

### 1. Clone & install

```bash
git clone https://github.com/yourname/medical-ai-system.git
cd medical-ai-system

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start the API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will start on **http://localhost:8000**

Interactive docs: **http://localhost:8000/docs**

### 3. Open the frontend

Simply open `frontend/index.html` in a browser.  
*(Or serve it: `python -m http.server 3000 --directory frontend`)*

---

## 🎓 Training Your Own Model <a name="training"></a>

### Step 1: Download dataset

```bash
# Example: Brain Tumor dataset from Kaggle
kaggle datasets download masoudnickparvar/brain-tumor-mri-dataset
unzip brain-tumor-mri-dataset.zip -d /data/brain_tumor
```

Your data directory should follow this layout:
```
/data/brain_tumor/
    train/
        glioma/       *.jpg
        meningioma/
        no_tumor/
        pituitary/
    val/
        glioma/
        …
```

### Step 2: Train

```bash
python training/train_model.py \
    --disease brain_tumor \
    --data_dir /data/brain_tumor \
    --output_dir model/
```

**Training phases:**
1. **Phase 1 (epochs 1–10):** Classifier head warm-up with frozen MobileNetV2 backbone
2. **Phase 2 (epochs 11–20):** Fine-tune top 40 backbone layers with LR=1e-5

Training outputs:
- `model/brain_tumor_model.keras` – final saved model
- `model/brain_tumor_best.keras` – best checkpoint by val_accuracy
- `model/brain_tumor_training_curves.png` – accuracy/loss plots

### Step 3: Evaluate

The training script prints:
- Validation accuracy, loss, AUC
- Per-class precision, recall, F1

Example output:
```
Val Accuracy: 0.9621
Val AUC:      0.9947

Classification Report:
              precision    recall  f1-score
glioma          0.97       0.96      0.96
meningioma      0.94       0.93      0.93
no_tumor        0.98       0.99      0.99
pituitary       0.97       0.98      0.97
```

---

## 🔌 API Reference <a name="api"></a>

### POST /predict

Upload a medical image for disease detection.

**Request (multipart/form-data):**
```
file         : binary  – JPEG or PNG medical image
disease_type : string  – "brain_tumor" | "pneumonia" | "skin_cancer"
```

**Response:**
```json
{
  "prediction": "glioma",
  "confidence": 0.923,
  "heatmap": "<base64-encoded PNG>",
  "all_probabilities": {
    "glioma": 0.923,
    "meningioma": 0.042,
    "no_tumor": 0.024,
    "pituitary": 0.011
  },
  "inference_time_seconds": 0.214,
  "disease_type": "brain_tumor"
}
```

**cURL example:**
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@mri_scan.jpg" \
  -F "disease_type=brain_tumor"
```

**Python example:**
```python
import requests

with open("scan.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": ("scan.jpg", f, "image/jpeg")},
        data={"disease_type": "brain_tumor"},
    )
print(response.json())
```

---

### GET /monitoring

Returns aggregated inference statistics.

```json
{
  "total_predictions": 42,
  "average_inference_time_seconds": 0.183,
  "average_confidence": 0.847,
  "min_confidence": 0.521,
  "max_confidence": 0.998,
  "prediction_distribution": {
    "glioma": 15,
    "no_tumor": 12,
    "pneumonia": 10
  },
  "recent_predictions": [...]
}
```

---

### GET /health

Liveness probe: `{"status": "healthy", "timestamp": "…"}`

### GET /diseases

Returns supported diseases, classes, and imaging modalities.

---

## 🎨 Frontend <a name="frontend"></a>

The React frontend (`frontend/index.html`) is a zero-build single file:

- **Analyse tab:** Select disease type → upload image → see prediction + Grad-CAM
- **Dashboard tab:** Live monitoring stats, charts, recent predictions table

No Node.js required – open directly in browser or serve with any static file server.

To build a full React app:
```bash
cd frontend
npm install
npm start   # dev server on :3000
npm run build  # production bundle
```

---

## 🐳 Docker Deployment <a name="docker"></a>

### Build and run (CPU)

```bash
# Build image
docker build -t medical-ai:latest -f docker/Dockerfile .

# Run container
docker run -d \
  --name medical-ai \
  -p 8000:8000 \
  -v $(pwd)/model:/app/model:ro \
  medical-ai:latest
```

### Run with GPU

```bash
docker run -d \
  --name medical-ai-gpu \
  --gpus all \
  -p 8000:8000 \
  medical-ai:latest
```

### Docker Compose (API + Frontend)

```bash
cd docker
docker compose up --build

# API:      http://localhost:8000
# Frontend: http://localhost:3000
```

### Useful Docker commands

```bash
# View logs
docker logs -f medical-ai

# Shell into container
docker exec -it medical-ai bash

# Health status
docker inspect --format='{{.State.Health.Status}}' medical-ai

# Stop
docker stop medical-ai && docker rm medical-ai
```

---

## ☁️ Cloud Deployment <a name="cloud"></a>

### AWS (ECS + ECR)

```bash
# 1. Push image to ECR
aws ecr create-repository --repository-name medical-ai
aws ecr get-login-password | docker login --username AWS --password-stdin <ECR_URI>
docker tag medical-ai:latest <ECR_URI>/medical-ai:latest
docker push <ECR_URI>/medical-ai:latest

# 2. Create ECS Fargate task
#    - vCPU: 2, Memory: 4GB (CPU inference)
#    - vCPU: 4, Memory: 16GB + GPU instance (GPU inference)
#    - Port mapping: 8000
#    - Health check: GET /health

# 3. Create ALB → Target Group → ECS Service
```

For GPU: Use `p3.2xlarge` EC2 instance with ECS EC2 launch type and `--gpus all` in container definition.

### Google Cloud Run

```bash
# 1. Build and push to Artifact Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/medical-ai

# 2. Deploy to Cloud Run (CPU)
gcloud run deploy medical-ai \
  --image gcr.io/PROJECT_ID/medical-ai \
  --platform managed \
  --port 8000 \
  --memory 4Gi \
  --cpu 2 \
  --allow-unauthenticated

# For GPU: Use Cloud Run GPU (T4 available)
# --gpu 1 --gpu-type nvidia-l4
```

### Render

1. Connect GitHub repo to Render
2. Create **Web Service**
3. Set **Docker** environment, **Dockerfile path:** `docker/Dockerfile`
4. Set env var: `TF_CPP_MIN_LOG_LEVEL=2`
5. Set port: `8000`
6. Deploy

### Railway

```bash
# Install Railway CLI
npm install -g @railway/cli
railway login
railway init

# Deploy
railway up

# Set environment variables
railway variables set TF_CPP_MIN_LOG_LEVEL=2
```

---

## ⚡ GPU Support <a name="gpu"></a>

The system automatically detects and uses GPU if available.

**Local GPU setup:**
```bash
# Install CUDA 12.x + cuDNN 8.x, then:
pip install tensorflow[and-cuda]==2.16.1

# Verify GPU detection
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**In training script:**
```bash
python training/train_model.py --disease brain_tumor --data_dir /data/brain_tumor
# Output: 🖥  GPU(s) detected: ['/device:GPU:0']
```

The training script enables GPU memory growth by default to prevent OOM errors.

---

## 📊 Model Monitoring <a name="monitoring"></a>

The built-in dashboard tracks:
- **Total predictions** since server start
- **Average inference time** (rolling 1000-request window)
- **Confidence statistics** (avg, min, max)
- **Prediction class distribution** (bar + pie charts)
- **Recent predictions** (last 50 with timestamps)

For production monitoring, replace the in-memory store with:
- **Redis** – fast in-memory store, supports TTLs
- **PostgreSQL** – persistent, queryable history
- **Prometheus + Grafana** – industry-standard metrics

---

## 📄 License

MIT License – free for research and commercial use.

---

## 🙏 Acknowledgements

- [MobileNetV2](https://arxiv.org/abs/1801.04381) – backbone architecture
- [Grad-CAM](https://arxiv.org/abs/1610.02391) – Selvaraju et al.
- [Kaggle](https://kaggle.com) – public medical imaging datasets
- [FastAPI](https://fastapi.tiangolo.com) – modern Python web framework
