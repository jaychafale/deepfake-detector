from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import io
import tempfile
import cv2

# Import your detectors
from deepfake_detector import DeepfakeDetector
from audio_detector import AudioDeepfakeDetector
from video_detector import VideoDeepfakeDetector
from utils import validate_image   # ✅ removed preprocess_image import (no longer used)

# FastAPI App
app = FastAPI(
    title="Deepfake Detection API",
    description="Image, Audio, and Video Deepfake Detection Service",
    version="1.0.0"
)

# CORS - allow any frontend to access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models once
image_detector = DeepfakeDetector()
audio_detector = AudioDeepfakeDetector()
video_detector = VideoDeepfakeDetector()


# -------------------------------
# ✅ ROOT
# -------------------------------
@app.get("/")
def root():
    return {
        "status": "running",
        "message": "Deepfake Detection API is live.",
        "endpoints": {
            "image": "/analyze/image",
            "audio": "/analyze/audio",
            "video": "/analyze/video"
        }
    }


# -------------------------------
# ✅ IMAGE DEEPFAKE DETECTION (FIXED)
# -------------------------------
@app.post("/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    try:
        # Read file
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")

        # Validate image
        if not validate_image(image):
            raise HTTPException(status_code=400, detail="Invalid or corrupted image")

        # ✅ FIX: Use ORIGINAL SIZE (no resize, no preprocess_image)
        img = np.asarray(image, dtype=np.float32) / 255.0
        img = np.expand_dims(img, axis=0)  # keep batch format

        # Run detection
        prediction, confidence = image_detector.predict(img)
        details = image_detector.get_analysis_details(img)

        return {
            "type": "image",
            "filename": file.filename,
            "prediction": prediction,
            "confidence": float(confidence),
            "details": details
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image analysis error: {str(e)}")


# -------------------------------
# ✅ AUDIO DEEPFAKE DETECTION
# -------------------------------
@app.post("/analyze/audio")
async def analyze_audio(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()

        # Your AudioDeepfakeDetector expects .read() method
        class TempFile:
            def __init__(self, data):
                self.data = data
            def read(self):
                return self.data

        temp_audio = TempFile(audio_bytes)

        prediction, confidence, details = audio_detector.analyze_audio(temp_audio)

        return {
            "type": "audio",
            "filename": file.filename,
            "prediction": prediction,
            "confidence": float(confidence),
            "details": details
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio analysis error: {str(e)}")


# -------------------------------
# ✅ VIDEO DEEPFAKE DETECTION
# -------------------------------
@app.post("/analyze/video")
async def analyze_video(file: UploadFile = File(...)):
    try:
        video_bytes = await file.read()

        class TempVideo:
            def __init__(self, data):
                self._data = data
            def read(self):
                return self._data

        temp_video = TempVideo(video_bytes)

        prediction, confidence, details = video_detector.analyze_video(temp_video)

        return {
            "type": "video",
            "filename": file.filename,
            "prediction": prediction,
            "confidence": float(confidence),
            "details": details
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video analysis error: {str(e)}")
