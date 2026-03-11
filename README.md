<div align="center">

# 🍅 TomatoGuard AI
### Tomato Leaf Disease Detection System

**An end-to-end AI-powered web application to detect diseases in tomato plants using YOLOv8 and FastAPI.**

[![React](https://img.shields.io/badge/React-19.2-61DAFB?logo=react&logoColor=white)](https://react.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-FF6B35?logo=python&logoColor=white)](https://ultralytics.com/)
[![Vite](https://img.shields.io/badge/Vite-7.x-646CFF?logo=vite&logoColor=white)](https://vite.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Disease Classes](#-disease-classes)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Training the Model](#1-train-the-yolov8-model-google-colab)
  - [Backend Setup](#2-backend-setup-fastapi)
  - [Frontend Setup](#3-frontend-setup-react--vite)
- [API Reference](#-api-reference)
- [How It Works](#-how-it-works)
- [Screenshots](#-screenshots)
- [Contributing](#-contributing)

---

## 🌱 Overview

TomatoGuard AI is a full-stack disease detection system built for farmers, agronomists, and researchers. It allows users to upload a photo of a tomato leaf and instantly receive:

- ✅ Disease diagnosis powered by a **YOLOv8** object detection model
- 💊 **Treatment recommendations** for each detected disease
- 📊 **Confidence scores** for each detection
- 🌿 Identification of **healthy** leaves

The system consists of:
- A **React/Vite frontend** with a premium dark-themed UI
- A **FastAPI Python backend** that loads and runs the trained YOLOv8 model
- A **Colab training notebook** to train your own model on the Kaggle tomato dataset

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔍 Real-time Detection | Upload any tomato leaf image and get results in seconds |
| 🤖 YOLOv8 Powered | Uses the latest Ultralytics YOLOv8 nano model |
| 💊 Treatment Advice | Each disease comes with expert treatment recommendations |
| 📈 Confidence Scores | See how confident the model is for each prediction |
| 🎨 Premium UI | Modern dark glassmorphism design with smooth animations |
| 📱 Responsive | Works on desktop and mobile browsers |
| 🔌 REST API | Clean FastAPI backend with auto-generated Swagger docs |
| 🐳 Docker Ready | Frontend includes a `Dockerfile` for containerized deployment |

---

## 🦠 Disease Classes

The model is trained to detect **7 disease categories**:

| # | Class Name | Disease ID | Severity | Treatment Summary |
|---|-----------|------------|----------|-------------------|
| 0 | Bacterial Spot | `bacterial_spot` | 🔴 High | Copper-based fungicides |
| 1 | Early Blight | `early_blight` | 🟡 Medium | Chlorothalonil or copper fungicides |
| 2 | Healthy | `healthy` | ✅ None | No treatment needed |
| 3 | Late Blight | `late_blight` | 🚨 Critical | Immediate fungicide, remove plants |
| 4 | Leaf Mold | `leaf_mold` | 🟢 Low | Reduce humidity, improve ventilation |
| 5 | Target Spot | `target_spot` | 🟡 Medium | Fungicides, avoid overhead watering |
| 6 | Black Spot | `black_spot` | 🔴 High | Copper/sulfur fungicides |

---

## 🛠️ Tech Stack

### Backend
- **[FastAPI](https://fastapi.tiangolo.com/)** — High-performance Python web framework
- **[Ultralytics YOLOv8](https://ultralytics.com/)** — State-of-the-art object detection
- **[Uvicorn](https://www.uvicorn.org/)** — ASGI server with auto-reload
- **[Pillow](https://python-pillow.org/)** — Image processing

### Frontend
- **[React 19](https://react.dev/)** — UI library
- **[Vite 7](https://vite.dev/)** — Lightning-fast dev server & bundler
- **[Lucide React](https://lucide.dev/)** — Beautiful icons
- **CSS3** — Glassmorphism, gradients, animations

### ML Training
- **[Google Colab](https://colab.research.google.com/)** — Free GPU training environment
- **[Kaggle Dataset](https://www.kaggle.com/datasets/farukalam/tomato-leaf-diseases-detection-computer-vision)** — Tomato leaf disease images

---

## 📁 Project Structure

```
Tomato_Leaf_Diseases_detection_using-ML/
│
├── 📓 tomato_yolov8_colab.ipynb    ← Google Colab training notebook
│
├── 📁 backend/                      ← FastAPI server
│   ├── main.py                      ← API app: /predict endpoint + YOLOv8 inference
│   ├── requirements.txt             ← Python dependencies
│   ├── README.md                    ← Backend-specific setup guide
│   └── model/
│       └── best.pt                  ← Trained YOLOv8 weights (not in git)
│
├── 📁 frontend/                     ← React/Vite application
│   ├── src/
│   │   ├── components/
│   │   │   ├── ImageUpload.jsx      ← Drag-and-drop image uploader
│   │   │   ├── Navbar.jsx           ← Navigation bar
│   │   │   └── PredictionResult.jsx ← Disease result cards
│   │   ├── pages/
│   │   │   ├── Home.jsx             ← Landing page
│   │   │   └── Detection.jsx        ← Detection page
│   │   ├── services/
│   │   │   └── api.js               ← FastAPI HTTP client
│   │   ├── constants/
│   │   │   └── diseases.js          ← Disease metadata (names, treatments, severity)
│   │   └── App.jsx                  ← Router setup
│   ├── vite.config.js               ← Vite config with /predict proxy
│   ├── package.json
│   └── Dockerfile                   ← Docker deployment config
│
├── .gitignore
└── README.md                        ← This file
```

---

## 🚀 Getting Started

### Prerequisites

Make sure you have the following installed:

| Tool | Version | Download |
|------|---------|----------|
| Python | 3.10+ | [python.org](https://python.org) |
| Node.js | 18+ | [nodejs.org](https://nodejs.org) |
| Git | Any | [git-scm.com](https://git-scm.com) |

---

### 1. Train the YOLOv8 Model (Google Colab)

> Skip this step if you already have a trained `best.pt` file.

1. Open the notebook in Google Colab:

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pranav-Darokar/Tomato_Leaf_Diseases_detection_using-ML/blob/main/tomato_yolov8_colab.ipynb)

2. Upload your **Kaggle API key** (`kaggle.json`) when prompted
3. The notebook will automatically:
   - Download the tomato disease dataset from Kaggle
   - Convert it to YOLO format
   - Train a `yolov8n` model
4. After training, download the model weights from:
   ```
   runs/detect/train/weights/best.pt
   ```
5. Place `best.pt` inside `backend/model/`:
   ```
   backend/model/best.pt
   ```

---

### 2. Backend Setup (FastAPI)

```bash
# Navigate to backend directory
cd backend

# Install Python dependencies
python -m pip install -r requirements.txt

# Start the FastAPI server
python main.py
```

The API will be running at: **http://localhost:8000**

✅ Verify it's working:
- Health check: http://localhost:8000/
- Interactive API docs: http://localhost:8000/docs

---

### 3. Frontend Setup (React + Vite)

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

The app will be running at: **http://localhost:5173**

> **Note:** The backend must be running on port `8000` before using the frontend. The Vite dev server is pre-configured to proxy `/predict` requests to `http://localhost:8000`.

---

## 📡 API Reference

### `GET /`
Health check — confirms the server and model are loaded.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "classes": ["Bacterial Spot", "Early_Blight", "Healthy", "Late_blight", "Leaf Mold", "Target_Spot", "black spot"]
}
```

---

### `POST /predict`
Upload a tomato leaf image for disease detection.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `file` — image file (JPEG, PNG, WEBP)

**Success Response:**
```json
{
  "success": true,
  "isHealthy": false,
  "diseases": [
    {
      "disease_name": "Early Blight",
      "disease_id": "early_blight",
      "confidence": 0.912,
      "treatment": "Apply fungicides containing chlorothalonil or copper. Improve air circulation.",
      "severity": "Medium"
    }
  ]
}
```

**Error Response:**
```json
{
  "success": false,
  "error": "No disease detected. Please upload a clear tomato leaf image."
}
```

**Test with curl:**
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@/path/to/tomato_leaf.jpg"
```

---

## ⚙️ How It Works

```
User uploads image
        │
        ▼
React frontend (localhost:5173)
        │  POST /predict (multipart/form-data)
        ▼
FastAPI backend (localhost:8000)
        │  PIL opens image → converts to RGB
        ▼
YOLOv8 model (best.pt)
        │  Runs object detection inference
        ▼
Class names mapped to disease metadata
(name, treatment, severity, confidence)
        │
        ▼
JSON response returned to frontend
        │
        ▼
PredictionResult component renders
disease cards with treatment & confidence
```

1. **User** drags or clicks to upload a tomato leaf photo on the Detection page
2. **Frontend** sends the image as a `multipart/form-data` POST to `http://localhost:8000/predict`
3. **FastAPI** reads the image using Pillow, runs it through the loaded YOLOv8 model
4. **YOLOv8** returns bounding boxes with class indices and confidence scores
5. **Backend** maps class names → disease metadata (treatment, severity) and returns JSON
6. **Frontend** displays result cards with the disease name, confidence, severity badge, and treatment advice

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

<div align="center">

Made with ❤️ by [Pranav Darokar](https://github.com/Pranav-Darokar)

⭐ Star this repo if it helped you!

</div>