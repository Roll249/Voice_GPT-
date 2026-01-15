#!/usr/bin/env python3
"""
CLI Setup Tool for Vietnamese Text-to-Speech Application
This script handles installation of dependencies and initial setup
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
import platform

def run_command(cmd, desc="Running command"):
    """Run a command and handle errors"""
    print(f"[INFO] {desc}")
    try:
        result = subprocess.run(
            cmd, shell=True, check=True, 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        print(f"[SUCCESS] {desc}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {desc}: {e.stderr}")
        return None

def check_gpu():
    """Check if CUDA is available"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[INFO] CUDA available: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("[WARNING] CUDA not available, using CPU (will be slower)")
            return False
    except ImportError:
        print("[WARNING] PyTorch not installed yet, will install")
        return False

def install_python_dependencies():
    """Install required Python packages"""
    print("[INFO] Installing Python dependencies...")
    
    packages = [
        "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
        "chatterbox-tts",
        "TTS",
        "ddddocr",
        "fastapi",
        "uvicorn",
        "gradio",
        "pydub",
        "librosa",
        "soundfile",
        "PyPDF2",
        "numpy",
        "scipy",
        "transformers",
        "requests",
        "python-multipart"
    ]
    
    for package in packages:
        cmd = f"pip install {package}"
        run_command(cmd, f"Installing {package}")

def install_ffmpeg():
    """Install FFmpeg based on OS"""
    system = platform.system().lower()
    
    if system == "linux":
        run_command("apt-get update && apt-get install -y ffmpeg", "Installing FFmpeg")
    elif system == "darwin":  # macOS
        run_command("brew install ffmpeg", "Installing FFmpeg")
    elif system == "windows":
        print("[INFO] Please install FFmpeg manually on Windows")
    else:
        print(f"[WARNING] Unsupported OS: {system}, please install FFmpeg manually")

def create_project_structure():
    """Create necessary directories"""
    dirs = [
        "models",
        "uploads",
        "outputs",
        "references",
        "temp"
    ]
    
    for dir_name in dirs:
        path = Path(dir_name)
        path.mkdir(exist_ok=True)
        print(f"[INFO] Created directory: {dir_name}")

def setup_backend():
    """Setup backend components"""
    print("[INFO] Setting up backend components...")
    
    # Create backend files
    backend_content = '''
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
import tempfile
import os
import torchaudio
import uuid
from pathlib import Path
from typing import Optional
import ddddocr
import PyPDF2
import re

app = FastAPI(title="Vietnamese TTS API")

# Global model instances (initialized on startup)
chatterbox_model = None
xtts_model = None

class VietnameseTTS:
    def __init__(self, model_type="chatterbox"):
        self.model_type = model_type
        self.reference_audio = None
        self.reference_path = None
        self.load_model()
    
    def load_model(self):
        """Load the TTS model"""
        global chatterbox_model, xtts_model
        
        if self.model_type == "chatterbox":
            if chatterbox_model is None:
                from chatterbox.tts import ChatterboxTTS
                chatterbox_model = ChatterboxTTS.from_pretrained(device="cuda" if self.has_cuda() else "cpu")
            self.model = chatterbox_model
            
        elif self.model_type == "xtts":
            if xtts_model is None:
                from TTS.api import TTS
                xtts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda" if self.has_cuda() else "cpu")
            self.model = xtts_model
    
    def has_cuda(self):
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def load_reference_voice(self, audio_path: str):
        """Load reference audio for voice cloning"""
        if self.model_type == "chatterbox":
            audio, sr = torchaudio.load(audio_path)
            
            # Resample if needed
            if hasattr(self.model, 'sr') and sr != self.model.sr:
                resampler = torchaudio.transforms.Resample(sr, self.model.sr)
                audio = resampler(audio)
                
            self.reference_audio = audio
            self.reference_path = audio_path
        elif self.model_type == "xtts":
            self.reference_path = audio_path
    
    def generate_speech(self, text: str, output_path: str, exaggeration: float = 0.5):
        """Generate speech with cloned voice"""
        if self.model_type == "chatterbox":
            if self.reference_audio is None:
                raise ValueError("No reference audio loaded")
                
            output = self.model.generate(
                text=text,
                audio_prompt=self.reference_audio,
                exaggeration=exaggeration
            )
            torchaudio.save(output_path, output, self.model.sr)
            
        elif self.model_type == "xtts":
            self.model.tts_to_file(
                text=text,
                speaker_wav=self.reference_path,
                language="vi",  # Vietnamese
                file_path=output_path
            )
        
        return output_path

# Global TTS instance
tts_instance = None

@app.on_event("startup")
async def startup_event():
    global tts_instance
    # Initialize with default model
    tts_instance = VietnameseTTS(model_type="chatterbox")
    print("TTS system initialized!")

@app.post("/upload-reference")
async def upload_reference(file: UploadFile = File(...)):
    """Upload reference audio for voice cloning"""
    try:
        # Save uploaded file temporarily
        temp_path = f"references/{uuid.uuid4()}_{file.filename}"
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Load reference audio
        tts_instance.load_reference_voice(temp_path)
        
        return {
            "status": "success", 
            "message": "Reference audio uploaded and loaded",
            "path": temp_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
async def generate_speech(
    text: str = Form(...),
    exaggeration: float = Form(0.5),
    model_type: str = Form("chatterbox")
):
    """Generate speech with cloned voice"""
    try:
        # Switch model if needed
        if tts_instance.model_type != model_type:
            tts_instance.__init__(model_type=model_type)
        
        # Generate output path
        output_path = f"outputs/{uuid.uuid4()}_output.wav"
        
        # Generate speech
        tts_instance.generate_speech(text, output_path, exaggeration)
        
        return FileResponse(
            output_path, 
            media_type="audio/wav", 
            filename=os.path.basename(output_path)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-pdf")
async def process_pdf(
    file: UploadFile = File(...),
    exaggeration: float = Form(0.5),
    model_type: str = Form("chatterbox")
):
    """Process PDF and generate audio"""
    try:
        # Save uploaded PDF
        temp_pdf_path = f"temp/{uuid.uuid4()}_{file.filename}"
        with open(temp_pdf_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Extract text from PDF
        text = extract_text_from_pdf(temp_pdf_path)
        
        # Clean text
        text = clean_text(text)
        
        # Split into chunks
        chunks = split_text_into_chunks(text)
        
        # Generate audio for each chunk
        output_files = []
        for i, chunk in enumerate(chunks):
            output_path = f"outputs/{uuid.uuid4()}_chunk_{i}.wav"
            tts_instance.generate_speech(chunk, output_path, exaggeration)
            output_files.append(output_path)
        
        # Combine all chunks
        final_output = f"outputs/{uuid.uuid4()}_full.wav"
        combine_audio_files(output_files, final_output)
        
        # Cleanup temporary files
        os.remove(temp_pdf_path)
        for chunk_file in output_files:
            os.remove(chunk_file)
        
        return FileResponse(
            final_output,
            media_type="audio/wav",
            filename="processed_document.wav"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF file"""
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\\n"
    return text

def clean_text(text: str) -> str:
    """Clean extracted text"""
    # Remove extra whitespace
    text = re.sub(r'\\s+', ' ', text)
    # Remove special characters but keep Vietnamese characters
    text = re.sub(r'[^\u00C0-\u1EF9\\w\\s.,!?\\'"-]', ' ', text)
    return text.strip()

def split_text_into_chunks(text: str, max_length: int = 200) -> list:
    """Split text into chunks for processing"""
    # Split by sentences
    sentences = re.split(r'(?<=[.!?])\\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_length:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def combine_audio_files(audio_paths: list, output_path: str):
    """Combine multiple audio files into one"""
    import torch
    all_audio = []
    
    for path in audio_paths:
        audio, sr = torchaudio.load(str(path))
        all_audio.append(audio)
        
        # Add small silence between chunks
        silence = torch.zeros(1, int(sr * 0.5))  # 0.5s silence
        all_audio.append(silence)
    
    combined = torch.cat(all_audio, dim=1)
    torchaudio.save(output_path, combined, sr)

@app.get("/")
async def root():
    return {"message": "Vietnamese TTS API is running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
    
    with open("src/backend/api_server.py", "w", encoding="utf-8") as f:
        f.write(backend_content)
    
    print("[INFO] Backend API server created")

def create_requirements():
    """Create requirements.txt file"""
    requirements = '''torch>=2.1.0
torchaudio>=2.1.0
chatterbox-tts>=0.1.0
TTS>=0.22.0
ddddocr>=1.0.0
fastapi>=0.100.0
uvicorn>=0.23.0
gradio>=4.0.0
pydub>=0.25.1
librosa>=0.10.0
soundfile>=0.12.1
PyPDF2>=3.0.0
numpy>=1.24.0
scipy>=1.11.0
transformers>=4.21.0
requests>=2.28.0
python-multipart>=0.0.6
'''
    
    with open("requirements.txt", "w") as f:
        f.write(requirements)
    
    print("[INFO] requirements.txt created")

def main():
    parser = argparse.ArgumentParser(description="Setup Vietnamese TTS Application")
    parser.add_argument("--install-deps", action="store_true", help="Install dependencies")
    parser.add_argument("--setup-backend", action="store_true", help="Setup backend components")
    parser.add_argument("--check-gpu", action="store_true", help="Check GPU availability")
    parser.add_argument("--all", action="store_true", help="Run all setup steps")
    
    args = parser.parse_args()
    
    print("Vietnamese Text-to-Speech Application Setup")
    print("=" * 50)
    
    # Create project structure
    create_project_structure()
    
    if args.all or args.check_gpu:
        check_gpu()
    
    if args.all or args.install_deps:
        install_ffmpeg()
        create_requirements()
        install_python_dependencies()
    
    if args.all or args.setup_backend:
        setup_backend()
    
    print("\n[SUCCESS] Setup completed!")
    print("\nTo run the application:")
    print("1. cd /workspace")
    print("2. uvicorn src.backend.api_server:app --reload --host 0.0.0.0 --port 8000")
    print("3. Open http://localhost:8000/docs for API documentation")

if __name__ == "__main__":
    main()