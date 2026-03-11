# TomatoGuard AI — FastAPI Backend

## Quick Start

### 1. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Ensure model is placed correctly
Your `best.pt` file must be at:
```
backend/model/best.pt
```
It was already copied here from your Downloads folder. If you retrain the model, just replace this file.

### 3. Run the server
```bash
# From project root:
cd backend
python main.py

# OR with uvicorn directly:
uvicorn main:app --reload --port 8000
```

The API will be available at: http://localhost:8000

### 4. API Endpoints
- `GET  /`        — Health check (also shows loaded model classes)
- `POST /predict` — Upload a tomato leaf image for disease detection

### Example curl test
```bash
curl -X POST http://localhost:8000/predict -F "file=@path/to/leaf.jpg"
```

### Docs
FastAPI auto-generates interactive docs at: http://localhost:8000/docs
