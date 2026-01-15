import os
import sys
import logging
from pathlib import Path
import tempfile
import shutil
from typing import Optional, List
import uuid

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import uvicorn
import torch
import ddddocr
from TTS.api import TTS
import librosa
import soundfile as sf
import numpy as np
from pydub import AudioSegment
import gradio as gr
import requests
from huggingface_hub import hf_hub_download
import zipfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Vietnamese Text-to-Speech API", version="1.0.0")

# Create necessary directories if they don't exist
directories = ["models", "uploads", "outputs", "references", "temp", "data"]
for directory in directories:
    os.makedirs(directory, exist_ok=True)

class TTSPayload(BaseModel):
    text: str
    model_type: str = "chatterbox"  # "chatterbox" or "xtts_v2"
    emotion_level: float = 1.0
    reference_audio_path: Optional[str] = None

class VoiceClonePayload(BaseModel):
    reference_audio_path: str
    output_model_path: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Vietnamese Text-to-Speech API"}

@app.post("/tts/")
async def text_to_speech(payload: TTSPayload):
    """
    Convert text to speech using either Chatterbox or XTTS-v2 model
    """
    try:
        logger.info(f"Received TTS request with model: {payload.model_type}")
        
        # Generate output filename
        output_filename = os.path.join("outputs", f"output_{uuid.uuid4().hex}.wav")
        
        if payload.model_type == "chatterbox":
            # Using Chatterbox TTS
            try:
                from chatterbox import ChatterboxTTS
                tts = ChatterboxTTS()
                
                # Generate speech with emotion
                tts.generate_speech(
                    payload.text,
                    output_filename,
                    emotion_level=payload.emotion_level
                )
            except ImportError:
                # Fallback to basic TTS if Chatterbox is not available
                logger.warning("Chatterbox not available, using fallback TTS")
                tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False, gpu=False)
                tts.tts_to_file(text=payload.text, speaker_wav=payload.reference_audio_path, file_path=output_filename)
        
        elif payload.model_type == "xtts_v2":
            # Using XTTS-v2 model
            try:
                # Load XTTS model
                tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False, gpu=False)
                
                # Generate speech with voice cloning if reference audio is provided
                if payload.reference_audio_path and os.path.exists(payload.reference_audio_path):
                    tts.tts_to_file(
                        text=payload.text,
                        speaker_wav=payload.reference_audio_path,
                        file_path=output_filename
                    )
                else:
                    # Without voice cloning
                    tts.tts_to_file(
                        text=payload.text,
                        file_path=output_filename
                    )
            except Exception as e:
                logger.error(f"Error with XTTS-v2: {str(e)}")
                raise HTTPException(status_code=500, detail=f"XTTS-v2 error: {str(e)}")
        
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported model type: {payload.model_type}")
        
        # Check if the file was created successfully
        if not os.path.exists(output_filename):
            raise HTTPException(status_code=500, detail="Failed to generate audio file")
        
        return JSONResponse(content={
            "status": "success",
            "output_file": output_filename,
            "message": "Audio generated successfully"
        })
    
    except Exception as e:
        logger.error(f"TTS Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

@app.post("/upload_reference/")
async def upload_reference_audio(file: UploadFile = File(...)):
    """
    Upload a reference audio file for voice cloning
    """
    try:
        # Create a unique filename
        unique_filename = f"reference_{uuid.uuid4().hex}_{file.filename}"
        file_location = os.path.join("references", unique_filename)
        
        # Save the uploaded file
        with open(file_location, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        return JSONResponse(content={
            "status": "success",
            "file_path": file_location,
            "message": "Reference audio uploaded successfully"
        })
    
    except Exception as e:
        logger.error(f"Upload Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/process_pdf/")
async def process_pdf(file: UploadFile = File(...)):
    """
    Process a PDF file and extract text using ddddocr
    """
    try:
        # Save the uploaded PDF temporarily
        temp_pdf_path = os.path.join("temp", f"temp_pdf_{uuid.uuid4().hex}.pdf")
        
        with open(temp_pdf_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # Use ddddocr to extract text from the PDF
        ocr = ddddocr.DdddOcr()
        
        # Since ddddocr primarily handles images, we'll use PyPDF2 for text extraction
        try:
            import PyPDF2
            with open(temp_pdf_path, 'rb') as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                text_content = ""
                for page in reader.pages:
                    text_content += page.extract_text() + "\n"
        except ImportError:
            # If PyPDF2 is not available, return an error
            raise HTTPException(status_code=500, detail="PyPDF2 library required for PDF text extraction")
        
        # Clean up temporary file
        os.remove(temp_pdf_path)
        
        return JSONResponse(content={
            "status": "success",
            "text": text_content,
            "message": "PDF processed successfully"
        })
    
    except Exception as e:
        logger.error(f"PDF Processing Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")

@app.get("/download/{filename:path}")
async def download_file(filename: str):
    """
    Download an output file
    """
    file_path = os.path.join("outputs", filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(path=file_path, filename=filename)

@app.post("/clone_voice/")
async def clone_voice(payload: VoiceClonePayload):
    """
    Clone a voice using reference audio
    """
    try:
        logger.info(f"Starting voice cloning process")
        
        # This is a simplified implementation
        # In a real scenario, this would involve training or fine-tuning a model
        
        # For now, we'll just validate the reference audio
        if not os.path.exists(payload.reference_audio_path):
            raise HTTPException(status_code=400, detail="Reference audio file not found")
        
        # Validate audio file
        try:
            audio, sr = librosa.load(payload.reference_audio_path)
            duration = librosa.get_duration(y=audio, sr=sr)
            logger.info(f"Reference audio duration: {duration} seconds")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid audio file: {str(e)}")
        
        # In a real implementation, this would train a model based on the reference audio
        # For now, we'll simulate the process by copying the reference to the output path
        # In a real scenario, this would be a trained model
        simulated_output_path = os.path.join("models", f"cloned_voice_{uuid.uuid4().hex}")
        os.makedirs(simulated_output_path, exist_ok=True)
        
        # Copy reference audio to the model directory
        shutil.copy2(payload.reference_audio_path, os.path.join(simulated_output_path, "reference.wav"))
        
        return JSONResponse(content={
            "status": "success",
            "model_path": simulated_output_path,
            "message": "Voice cloning simulation completed (in a real implementation, this would train a model)"
        })
    
    except Exception as e:
        logger.error(f"Voice Cloning Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Voice cloning failed: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy", "message": "API is running"}

if __name__ == "__main__":
    uvicorn.run(
        "src.backend.api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )