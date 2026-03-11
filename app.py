from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
from pathlib import Path
import shutil
from datetime import datetime
import cv2
import torch
import numpy as np
from PIL import Image
from io import BytesIO

app = FastAPI(title="Deepfake Detection System", version="1.0")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create upload directories
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Simple detection function (using basic features)
def detect_deepfake_image(image_path: str) -> dict:
    img = cv2.imread(image_path)
    if img is None:
        return {"error": "Could not read image"}
    
    # Basic detection using face detection + texture analysis
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        return {
            "is_deepfake": False,
            "confidence": 0.2,
            "reason": "No faces detected"
        }
    
    # Simple heuristic based on texture consistency
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    is_deepfake = laplacian_var < 100
    confidence = min(0.95, max(0.5, (100 - laplacian_var) / 100))
    
    return {
        "is_deepfake": is_deepfake,
        "confidence": float(confidence),
        "faces_detected": len(faces),
        "reason": "Based on texture consistency analysis"
    }

def detect_deepfake_video(video_path: str, sample_frames: int = 10) -> dict:
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if total_frames == 0:
        return {"error": "Could not read video"}
    
    frame_indices = np.linspace(0, total_frames - 1, min(sample_frames, total_frames), dtype=int)
    deepfake_scores = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        score = min(0.95, max(0.5, (100 - laplacian_var) / 100))
        deepfake_scores.append(score)
    
    cap.release()
    
    if not deepfake_scores:
        return {"error": "No frames analyzed"}
    
    avg_confidence = float(np.mean(deepfake_scores))
    is_deepfake = avg_confidence > 0.6
    
    return {
        "is_deepfake": is_deepfake,
        "average_confidence": avg_confidence,
        "frames_analyzed": len(deepfake_scores),
        "duration_seconds": total_frames / fps if fps > 0 else 0
    }

@app.get("/", response_class=HTMLResponse)
async def get_ui():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Deepfake Detection</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            .container {
                background: rgba(0,0,0,0.5);
                border-radius: 10px;
                padding: 30px;
            }
            h1 { text-align: center; margin-bottom: 30px; }
            .upload-box {
                border: 2px dashed white;
                padding: 30px;
                text-align: center;
                border-radius: 10px;
                margin: 20px 0;
            }
            input[type="file"] {
                display: block;
                margin: 20px auto;
                padding: 10px;
                cursor: pointer;
            }
            button {
                background: #667eea;
                color: white;
                border: none;
                padding: 12px 30px;
                font-size: 16px;
                border-radius: 5px;
                cursor: pointer;
                margin: 10px;
            }
            button:hover { background: #764ba2; }
            #result {
                margin-top: 20px;
                padding: 15px;
                background: rgba(255,255,255,0.1);
                border-radius: 5px;
                display: none;
            }
            .loading { text-align: center; display: none; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎬 Deepfake Detection</h1>
            
            <div class="upload-box">
                <input type="file" id="fileInput" accept="image/*,video/*" />
                <button onclick="detectFile()">🔍 Detect Deepfake</button>
            </div>
            
            <div class="loading" id="loading">⏳ Processing...</div>
            
            <div id="result">
                <h3>Results:</h3>
                <p id="resultText"></p>
            </div>
        </div>
        
        <script>
            async function detectFile() {
                const file = document.getElementById('fileInput').files[0];
                if (!file) {
                    alert('Please select a file');
                    return;
                }
                
                const loading = document.getElementById('loading');
                const result = document.getElementById('result');
                loading.style.display = 'block';
                result.style.display = 'none';
                
                try {
                    const formData = new FormData();
                    formData.append('file', file);
                    
                    const endpoint = file.type.startsWith('image/') ? '/api/image/detect' : '/api/video/detect';
                    const response = await fetch(endpoint, {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    loading.style.display = 'none';
                    result.style.display = 'block';
                    
                    let text = `<strong>Deepfake Detected: ${data.is_deepfake ? '✅ YES' : '❌ NO'}</strong>
                    <br>Confidence: ${(data.confidence * 100).toFixed(2)}%`;
                    
                    if (data.frames_analyzed) {
                        text += `<br>Frames Analyzed: ${data.frames_analyzed}`;
                    }
                    
                    document.getElementById('resultText').innerHTML = text;
                } catch (error) {
                    loading.style.display = 'none';
                    result.style.display = 'block';
                    document.getElementById('resultText').innerHTML = `Error: ${error.message}`;
                }
            }
        </script>
    </body>
    </html>
    """

@app.post("/api/image/detect")
async def detect_image(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        raise HTTPException(status_code=400, detail="Invalid image format")
    
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    result = detect_deepfake_image(str(file_path))
    os.remove(file_path)
    return result

@app.post("/api/video/detect")
async def detect_video(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        raise HTTPException(status_code=400, detail="Invalid video format")
    
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    result = detect_deepfake_video(str(file_path))
    os.remove(file_path)
    return result

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)