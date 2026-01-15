import os
import sys
import logging
from pathlib import Path
import tempfile
import shutil
from typing import Optional
import uuid

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydub import AudioSegment
import uvicorn
import numpy as np
import PyPDF2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Vietnamese Text-to-Speech API (Simplified)", version="1.0.0")

# Create necessary directories if they don't exist
directories = ["models", "uploads", "outputs", "references", "temp", "data"]
for directory in directories:
    os.makedirs(directory, exist_ok=True)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Vietnamese Text-to-Speech API (Simplified)"}

@app.post("/tts/")
async def text_to_speech(text: str, model_type: str = "basic"):
    """
    Simplified TTS endpoint - creates a placeholder audio file
    In a real implementation, this would generate actual speech
    """
    try:
        logger.info(f"Received TTS request with model: {model_type}")
        
        # Generate output filename
        output_filename = os.path.join("outputs", f"output_{uuid.uuid4().hex}.wav")
        
        # Create a simple placeholder audio file (silence)
        # In a real implementation, this would generate actual speech
        duration_ms = len(text) * 50  # Approximate duration based on text length
        silence = AudioSegment.silent(duration=duration_ms, frame_rate=22050)
        silence.export(output_filename, format="wav")
        
        return JSONResponse(content={
            "status": "success",
            "output_file": output_filename,
            "message": "Placeholder audio generated successfully - in a full implementation this would generate actual speech"
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
    Process a PDF file and extract text
    """
    try:
        # Save the uploaded PDF temporarily
        temp_pdf_path = os.path.join("temp", f"temp_pdf_{uuid.uuid4().hex}.pdf")
        
        with open(temp_pdf_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # Extract text from the PDF
        try:
            with open(temp_pdf_path, 'rb') as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                text_content = ""
                for page in reader.pages:
                    text_content += page.extract_text() + "\n"
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"PDF processing error: {str(e)}")
        
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

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy", "message": "API is running"}

if __name__ == "__main__":
    uvicorn.run(
        "src.backend.simple_api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )