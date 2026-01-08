# Hướng Dẫn Cài Đặt & Sử Dụng Voice Cloning

## Mục Lục
1. [Yêu Cầu Hệ Thống](#1-yêu-cầu-hệ-thống)
2. [Cài Đặt Chatterbox](#2-cài-đặt-chatterbox)
3. [Cài Đặt GPT-SoVITS](#3-cài-đặt-gpt-sovits)
4. [Cài Đặt XTTS-v2](#4-cài-đặt-xtts-v2)
5. [Chuẩn Bị Audio Reference](#5-chuẩn-bị-audio-reference)
6. [Workflow Sử Dụng](#6-workflow-sử-dụng)
7. [Tích Hợp Vào Ứng Dụng](#7-tích-hợp-vào-ứng-dụng)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Yêu Cầu Hệ Thống

### 1.1 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | NVIDIA GTX 1060 6GB | RTX 3080/4080 16GB+ |
| **VRAM** | 6GB | 16GB+ |
| **RAM** | 16GB | 32GB |
| **Storage** | 20GB free | 50GB+ SSD |
| **CPU** | Intel i5/AMD Ryzen 5 | Intel i7/AMD Ryzen 7 |

### 1.2 Software Requirements

| Software | Version |
|----------|---------|
| Python | 3.10.x (recommended) |
| CUDA | 11.8 hoặc 12.1 |
| cuDNN | 8.6+ |
| Git | Latest |
| FFmpeg | Latest |

### 1.3 Kiểm Tra GPU

```bash
# Kiểm tra NVIDIA GPU
nvidia-smi

# Output mong đợi:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 535.xx       Driver Version: 535.xx       CUDA Version: 12.2     |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# |===============================+======================+======================|
# |   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0  On |                  N/A |
# | 30%   45C    P8    15W / 320W |   1234MiB / 16384MiB |      0%      Default |
# +-------------------------------+----------------------+----------------------+
```

### 1.4 Cài Đặt Python 3.10

#### Windows
```powershell
# Download từ python.org hoặc dùng winget
winget install Python.Python.3.10

# Verify
python --version
# Output: Python 3.10.x
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev
```

### 1.5 Cài Đặt CUDA Toolkit

#### Windows
1. Download CUDA Toolkit 11.8 từ: https://developer.nvidia.com/cuda-11-8-0-download-archive
2. Chạy installer, chọn Express Installation
3. Restart máy

#### Linux
```bash
# CUDA 11.8
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# Add to PATH
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 1.6 Cài Đặt FFmpeg

#### Windows
```powershell
# Dùng winget
winget install FFmpeg

# Hoặc dùng Chocolatey
choco install ffmpeg

# Verify
ffmpeg -version
```

#### Linux
```bash
sudo apt install ffmpeg
```

---

## 2. Cài Đặt Chatterbox

### 2.1 Tạo Virtual Environment

```bash
# Tạo thư mục project
mkdir voice-cloning && cd voice-cloning

# Tạo virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

### 2.2 Cài Đặt PyTorch với CUDA

```bash
# CUDA 11.8
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
# Output: CUDA available: True
```

### 2.3 Cài Đặt Chatterbox

```bash
pip install chatterbox-tts
```

### 2.4 Download Model (Tự động khi chạy lần đầu)

```python
# test_chatterbox.py
from chatterbox.tts import ChatterboxTTS

# Model sẽ tự động download (~2-3GB)
model = ChatterboxTTS.from_pretrained(device="cuda")
print("Model loaded successfully!")
```

### 2.5 Test Basic Usage

```python
# basic_test.py
import torchaudio
from chatterbox.tts import ChatterboxTTS

# Load model
model = ChatterboxTTS.from_pretrained(device="cuda")

# Generate với default voice
output = model.generate(
    text="Hello, this is a test of the Chatterbox text to speech system."
)

# Save output
torchaudio.save("test_output.wav", output, model.sr)
print("Audio saved to test_output.wav")
```

### 2.6 Voice Cloning với Chatterbox

```python
# voice_clone.py
import torchaudio
from chatterbox.tts import ChatterboxTTS

def clone_voice(reference_audio_path, text, output_path, exaggeration=0.5):
    """
    Clone voice từ reference audio và generate speech.

    Args:
        reference_audio_path: Path đến file audio reference (wav, mp3)
        text: Text cần đọc
        output_path: Path lưu output
        exaggeration: Mức độ emotion (0.0=monotone, 1.0=expressive)
    """
    # Load model
    model = ChatterboxTTS.from_pretrained(device="cuda")

    # Load reference audio
    audio_prompt, sr = torchaudio.load(reference_audio_path)

    # Resample nếu cần
    if sr != model.sr:
        resampler = torchaudio.transforms.Resample(sr, model.sr)
        audio_prompt = resampler(audio_prompt)

    # Generate với voice cloning
    output = model.generate(
        text=text,
        audio_prompt=audio_prompt,
        exaggeration=exaggeration
    )

    # Save
    torchaudio.save(output_path, output, model.sr)
    print(f"Audio saved to {output_path}")
    return output_path

# Usage
if __name__ == "__main__":
    clone_voice(
        reference_audio_path="reference_voice.wav",
        text="This is my cloned voice speaking English.",
        output_path="cloned_output.wav",
        exaggeration=0.5
    )
```

### 2.7 Batch Processing

```python
# batch_clone.py
import torchaudio
from chatterbox.tts import ChatterboxTTS
from pathlib import Path

class VoiceCloner:
    def __init__(self, device="cuda"):
        self.model = ChatterboxTTS.from_pretrained(device=device)
        self.reference_audio = None

    def load_reference(self, audio_path):
        """Load reference audio một lần để dùng cho nhiều generation."""
        audio, sr = torchaudio.load(audio_path)
        if sr != self.model.sr:
            resampler = torchaudio.transforms.Resample(sr, self.model.sr)
            audio = resampler(audio)
        self.reference_audio = audio
        print(f"Loaded reference: {audio_path}")

    def generate(self, text, output_path, exaggeration=0.5):
        """Generate speech với reference đã load."""
        if self.reference_audio is None:
            raise ValueError("Please load reference audio first!")

        output = self.model.generate(
            text=text,
            audio_prompt=self.reference_audio,
            exaggeration=exaggeration
        )
        torchaudio.save(output_path, output, self.model.sr)
        return output_path

    def batch_generate(self, texts, output_dir, exaggeration=0.5):
        """Generate nhiều audio từ list texts."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        outputs = []
        for i, text in enumerate(texts):
            output_path = output_dir / f"output_{i:03d}.wav"
            self.generate(text, str(output_path), exaggeration)
            outputs.append(output_path)
            print(f"Generated {i+1}/{len(texts)}")

        return outputs

# Usage
if __name__ == "__main__":
    cloner = VoiceCloner()
    cloner.load_reference("reference_voice.wav")

    texts = [
        "Hello, welcome to the English learning system.",
        "Today we will practice pronunciation.",
        "Repeat after me: The quick brown fox jumps over the lazy dog."
    ]

    outputs = cloner.batch_generate(texts, "outputs/")
    print(f"Generated {len(outputs)} audio files")
```

---

## 3. Cài Đặt GPT-SoVITS

### 3.1 Clone Repository

```bash
# Clone repo
git clone https://github.com/RVC-Boss/GPT-SoVITS.git
cd GPT-SoVITS

# Tạo virtual environment
python -m venv venv

# Activate
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3.2 Cài Đặt Dependencies

```bash
# Install PyTorch (CUDA 11.8)
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install requirements
pip install -r requirements.txt
```

### 3.3 Download Pre-trained Models

```bash
# Tạo thư mục
mkdir -p GPT_SoVITS/pretrained_models

# Download từ Hugging Face (tự động trong WebUI)
# Hoặc manual download từ:
# https://huggingface.co/lj1995/GPT-SoVITS/tree/main
```

### 3.4 Chạy WebUI

```bash
# Windows
python webui.py

# Linux
python webui.py

# Mở browser: http://localhost:9874
```

### 3.5 Training Workflow (WebUI)

```
Bước 1: Upload Audio
├── Tab "1-GPT-SoVITS-TTS"
├── Upload audio files (1-5 phút clean audio)
└── Hỗ trợ: wav, mp3, flac

Bước 2: Audio Processing
├── Click "1A-训练集格式化一键处理" (One-click format)
├── Tự động: slice, denoise, ASR transcription
└── Kiểm tra output trong logs

Bước 3: SoVITS Training
├── Tab "1B-微调训练-SoVITS"
├── Chọn config và epochs (default OK)
├── Click "开启SoVITS训练" (Start Training)
└── Wait 15-30 phút

Bước 4: GPT Training
├── Tab "1C-微调训练-GPT"
├── Chọn config
├── Click "开启GPT训练" (Start Training)
└── Wait 15-30 phút

Bước 5: Inference
├── Tab "1-推理"
├── Load trained models
├── Input text
├── Generate!
```

### 3.6 Inference API (Code)

```python
# gpt_sovits_api.py
import sys
sys.path.append("GPT-SoVITS")

from inference_main import inference

def generate_speech(
    text,
    ref_audio_path,
    ref_text,
    output_path,
    gpt_model_path,
    sovits_model_path
):
    """
    Generate speech với GPT-SoVITS.

    Args:
        text: Text cần đọc
        ref_audio_path: Audio reference (3-10 giây)
        ref_text: Transcript của reference audio
        output_path: Output file path
        gpt_model_path: Path đến GPT model đã train
        sovits_model_path: Path đến SoVITS model đã train
    """
    # Load và inference
    result = inference(
        text=text,
        ref_audio_path=ref_audio_path,
        ref_text=ref_text,
        gpt_model_path=gpt_model_path,
        sovits_model_path=sovits_model_path,
        output_path=output_path
    )
    return result

# Usage
if __name__ == "__main__":
    generate_speech(
        text="Hello, this is a test.",
        ref_audio_path="reference.wav",
        ref_text="This is the reference audio transcript.",
        output_path="output.wav",
        gpt_model_path="trained_models/gpt_model.ckpt",
        sovits_model_path="trained_models/sovits_model.pth"
    )
```

---

## 4. Cài Đặt XTTS-v2

### 4.1 Cài Đặt

```bash
# Tạo environment
python -m venv venv
source venv/bin/activate  # hoặc .\venv\Scripts\activate trên Windows

# Install PyTorch
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install Coqui TTS
pip install TTS
```

### 4.2 Verify Installation

```bash
# List available models
tts --list_models

# Tìm XTTS-v2
# tts_models/multilingual/multi-dataset/xtts_v2
```

### 4.3 Basic Usage (CLI)

```bash
# Generate với voice cloning
tts --model_name tts_models/multilingual/multi-dataset/xtts_v2 \
    --text "Hello, this is a voice cloning test." \
    --speaker_wav reference.wav \
    --language_idx en \
    --out_path output.wav
```

### 4.4 Python API

```python
# xtts_clone.py
from TTS.api import TTS
import torch

class XTTSCloner:
    def __init__(self, device="cuda"):
        self.device = device
        print("Loading XTTS-v2 model...")
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        print("Model loaded!")

    def clone_and_speak(self, text, speaker_wav, language="en", output_path="output.wav"):
        """
        Clone voice và generate speech.

        Args:
            text: Text cần đọc
            speaker_wav: Path đến file audio reference
            language: Mã ngôn ngữ (en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, ja, hu, ko, hi)
            output_path: Path lưu output
        """
        self.tts.tts_to_file(
            text=text,
            speaker_wav=speaker_wav,
            language=language,
            file_path=output_path
        )
        print(f"Audio saved to {output_path}")
        return output_path

    def batch_generate(self, texts, speaker_wav, language="en", output_dir="outputs"):
        """Generate nhiều audio files."""
        import os
        os.makedirs(output_dir, exist_ok=True)

        outputs = []
        for i, text in enumerate(texts):
            output_path = os.path.join(output_dir, f"output_{i:03d}.wav")
            self.clone_and_speak(text, speaker_wav, language, output_path)
            outputs.append(output_path)
            print(f"Generated {i+1}/{len(texts)}")

        return outputs

# Usage
if __name__ == "__main__":
    cloner = XTTSCloner()

    # Single generation
    cloner.clone_and_speak(
        text="Hello, this is a test of XTTS voice cloning.",
        speaker_wav="reference.wav",
        language="en",
        output_path="output.wav"
    )

    # Batch generation
    texts = [
        "Welcome to the English learning program.",
        "Today we will practice pronunciation.",
        "Please repeat after me."
    ]
    cloner.batch_generate(texts, "reference.wav", "en", "outputs/")
```

### 4.5 Streaming Generation

```python
# xtts_streaming.py
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import torch
import torchaudio

class XTTSStreamer:
    def __init__(self, device="cuda"):
        print("Loading XTTS model...")
        config = XttsConfig()
        config.load_json("path/to/config.json")

        self.model = Xtts.init_from_config(config)
        self.model.load_checkpoint(config, checkpoint_dir="path/to/checkpoint/")
        self.model.to(device)
        self.device = device
        print("Model loaded!")

    def get_conditioning(self, speaker_wav):
        """Extract speaker conditioning từ reference audio."""
        gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(
            audio_path=speaker_wav
        )
        return gpt_cond_latent, speaker_embedding

    def stream_generate(self, text, gpt_cond_latent, speaker_embedding, language="en"):
        """Stream generation với low latency."""
        chunks = self.model.inference_stream(
            text,
            language,
            gpt_cond_latent,
            speaker_embedding,
            enable_text_splitting=True
        )

        for chunk in chunks:
            yield chunk

    def generate_with_streaming(self, text, speaker_wav, language="en", output_path="output.wav"):
        """Generate với streaming và save."""
        gpt_cond_latent, speaker_embedding = self.get_conditioning(speaker_wav)

        all_chunks = []
        for chunk in self.stream_generate(text, gpt_cond_latent, speaker_embedding, language):
            all_chunks.append(chunk)
            # Có thể play chunk ở đây cho realtime playback

        # Concatenate và save
        full_audio = torch.cat(all_chunks, dim=0)
        torchaudio.save(output_path, full_audio.unsqueeze(0), 24000)
        return output_path
```

---

## 5. Chuẩn Bị Audio Reference

### 5.1 Yêu Cầu Audio Quality

| Tiêu chí | Yêu cầu | Lý do |
|----------|---------|-------|
| **Độ dài** | 5-60 giây | Quá ngắn: thiếu info, quá dài: không cần thiết |
| **Sample rate** | 22050Hz+ | Standard cho TTS |
| **Channels** | Mono preferred | Stereo OK nhưng sẽ convert |
| **Format** | WAV, MP3, FLAC | WAV tốt nhất |
| **Noise** | Minimal | Background noise ảnh hưởng clone quality |
| **Content** | Clear speech | Không có music, sound effects |

### 5.2 Audio Preprocessing

```python
# audio_preprocess.py
import subprocess
import os
from pathlib import Path

def preprocess_audio(input_path, output_path, target_sr=22050):
    """
    Preprocess audio cho voice cloning.
    - Convert to mono
    - Resample to target sample rate
    - Normalize volume
    - Remove silence
    """

    # FFmpeg command
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ac", "1",  # Mono
        "-ar", str(target_sr),  # Sample rate
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11,silenceremove=start_periods=1:start_silence=0.1:start_threshold=-50dB",
        output_path
    ]

    subprocess.run(cmd, capture_output=True)
    print(f"Preprocessed: {input_path} -> {output_path}")
    return output_path

def batch_preprocess(input_dir, output_dir, target_sr=22050):
    """Preprocess tất cả audio files trong directory."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_extensions = [".wav", ".mp3", ".flac", ".m4a", ".ogg"]

    for file in input_dir.iterdir():
        if file.suffix.lower() in audio_extensions:
            output_path = output_dir / f"{file.stem}_processed.wav"
            preprocess_audio(str(file), str(output_path), target_sr)

# Usage
if __name__ == "__main__":
    # Single file
    preprocess_audio("raw_audio.mp3", "processed_audio.wav")

    # Batch
    batch_preprocess("raw_audios/", "processed_audios/")
```

### 5.3 Noise Removal với Denoiser

```python
# denoise.py
import torch
import torchaudio
from denoiser import pretrained
from denoiser.dsp import convert_audio

def denoise_audio(input_path, output_path):
    """Remove background noise từ audio."""
    # Load pretrained denoiser
    model = pretrained.dns64()
    model.eval()

    # Load audio
    wav, sr = torchaudio.load(input_path)

    # Convert to model's expected format
    wav = convert_audio(wav, sr, model.sample_rate, model.chin)

    # Denoise
    with torch.no_grad():
        denoised = model(wav[None])[0]

    # Save
    torchaudio.save(output_path, denoised, model.sample_rate)
    print(f"Denoised: {input_path} -> {output_path}")
    return output_path

# Install: pip install denoiser
```

### 5.4 Audio Slicing

```python
# audio_slice.py
from pydub import AudioSegment
from pydub.silence import split_on_silence
from pathlib import Path

def slice_audio(input_path, output_dir, min_silence_len=500, silence_thresh=-40, min_length=3000, max_length=15000):
    """
    Slice audio thành các clips dựa trên silence.

    Args:
        input_path: Path đến audio file
        output_dir: Directory lưu clips
        min_silence_len: Minimum silence length (ms) để split
        silence_thresh: Silence threshold (dB)
        min_length: Minimum clip length (ms)
        max_length: Maximum clip length (ms)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load audio
    audio = AudioSegment.from_file(input_path)

    # Split on silence
    chunks = split_on_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=200  # Keep 200ms silence at edges
    )

    # Filter và save
    saved_clips = []
    for i, chunk in enumerate(chunks):
        # Skip too short or too long
        if len(chunk) < min_length or len(chunk) > max_length:
            continue

        output_path = output_dir / f"clip_{i:03d}.wav"
        chunk.export(output_path, format="wav")
        saved_clips.append(output_path)
        print(f"Saved: {output_path} ({len(chunk)/1000:.1f}s)")

    print(f"Total clips: {len(saved_clips)}")
    return saved_clips

# Install: pip install pydub

# Usage
if __name__ == "__main__":
    slice_audio("long_audio.mp3", "clips/")
```

### 5.5 Combine Multiple References

```python
# combine_references.py
import torchaudio
import torch
from pathlib import Path

def combine_references(audio_paths, output_path, max_duration=60):
    """
    Combine multiple audio files thành một reference.

    Args:
        audio_paths: List các paths đến audio files
        output_path: Output file path
        max_duration: Maximum duration (seconds)
    """
    all_audio = []
    total_duration = 0
    target_sr = 22050

    for path in audio_paths:
        audio, sr = torchaudio.load(path)

        # Resample nếu cần
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            audio = resampler(audio)

        # Convert to mono
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        duration = audio.shape[1] / target_sr

        if total_duration + duration > max_duration:
            # Trim to fit
            remaining = max_duration - total_duration
            samples_to_take = int(remaining * target_sr)
            audio = audio[:, :samples_to_take]
            all_audio.append(audio)
            break

        all_audio.append(audio)
        total_duration += duration

    # Concatenate
    combined = torch.cat(all_audio, dim=1)

    # Save
    torchaudio.save(output_path, combined, target_sr)
    print(f"Combined {len(all_audio)} files -> {output_path} ({combined.shape[1]/target_sr:.1f}s)")
    return output_path

# Usage
if __name__ == "__main__":
    audio_files = list(Path("clips/").glob("*.wav"))
    combine_references(audio_files, "combined_reference.wav")
```

---

## 6. Workflow Sử Dụng

### 6.1 Complete Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                      COMPLETE WORKFLOW                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Step 1: Collect Raw Audio                                      │
│  ├── Source: YouTube, podcast, recordings                       │
│  ├── Total: 1-5 phút clean speech                              │
│  └── Format: Any audio format                                   │
│                           │                                      │
│                           ▼                                      │
│  Step 2: Preprocess                                             │
│  ├── Convert to WAV/mono/22050Hz                               │
│  ├── Denoise (optional)                                        │
│  └── Slice into clips (5-15s each)                             │
│                           │                                      │
│                           ▼                                      │
│  Step 3: Voice Cloning                                          │
│  ├── Option A: Zero-shot (Chatterbox/XTTS)                     │
│  │   └── Just provide reference audio                          │
│  └── Option B: Fine-tune (GPT-SoVITS)                          │
│       └── Train on clips (30-60 min)                           │
│                           │                                      │
│                           ▼                                      │
│  Step 4: Text-to-Speech                                         │
│  ├── Input: PDF/Text content                                   │
│  ├── Process: Extract text, clean                              │
│  └── Generate: TTS with cloned voice                           │
│                           │                                      │
│                           ▼                                      │
│  Step 5: Output                                                 │
│  ├── Format: WAV/MP3                                           │
│  └── Use: Learning, playback                                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Full Example Code

```python
# full_pipeline.py
import os
from pathlib import Path
import torchaudio
from chatterbox.tts import ChatterboxTTS
import PyPDF2
import re

class VoiceLearningSystem:
    def __init__(self, model_type="chatterbox", device="cuda"):
        """
        Initialize Voice Learning System.

        Args:
            model_type: "chatterbox", "xtts", hoặc "gpt-sovits"
            device: "cuda" hoặc "cpu"
        """
        self.device = device
        self.model_type = model_type
        self.model = None
        self.reference_audio = None

        self._load_model()

    def _load_model(self):
        """Load TTS model."""
        print(f"Loading {self.model_type} model...")

        if self.model_type == "chatterbox":
            self.model = ChatterboxTTS.from_pretrained(device=self.device)
        elif self.model_type == "xtts":
            from TTS.api import TTS
            self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        print("Model loaded!")

    def load_reference_voice(self, audio_path):
        """Load reference audio for voice cloning."""
        print(f"Loading reference: {audio_path}")

        audio, sr = torchaudio.load(audio_path)

        # Resample if needed (Chatterbox expects 24000Hz)
        if self.model_type == "chatterbox" and sr != self.model.sr:
            resampler = torchaudio.transforms.Resample(sr, self.model.sr)
            audio = resampler(audio)

        self.reference_audio = audio
        self.reference_path = audio_path
        print("Reference voice loaded!")

    def extract_text_from_pdf(self, pdf_path):
        """Extract text từ PDF file."""
        text = ""
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"

        # Clean text
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces -> single
        text = re.sub(r'[^\w\s.,!?\'"-]', '', text)  # Remove special chars
        return text.strip()

    def split_text_into_chunks(self, text, max_length=200):
        """Split text thành chunks nhỏ để generate."""
        # Split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)

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

    def generate_speech(self, text, output_path, exaggeration=0.5):
        """Generate speech với cloned voice."""
        if self.reference_audio is None:
            raise ValueError("Please load reference voice first!")

        if self.model_type == "chatterbox":
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
                language="en",
                file_path=output_path
            )

        return output_path

    def process_document(self, input_path, output_dir, exaggeration=0.5):
        """
        Process document (PDF hoặc text) và generate audio.

        Args:
            input_path: Path đến PDF hoặc text file
            output_dir: Directory lưu output audio files
            exaggeration: Emotion level (0.0-1.0)

        Returns:
            List of generated audio paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract text
        if input_path.endswith('.pdf'):
            text = self.extract_text_from_pdf(input_path)
        else:
            with open(input_path, 'r', encoding='utf-8') as f:
                text = f.read()

        print(f"Extracted text: {len(text)} characters")

        # Split into chunks
        chunks = self.split_text_into_chunks(text)
        print(f"Split into {len(chunks)} chunks")

        # Generate audio for each chunk
        outputs = []
        for i, chunk in enumerate(chunks):
            output_path = output_dir / f"chunk_{i:03d}.wav"
            print(f"Generating {i+1}/{len(chunks)}: {chunk[:50]}...")

            self.generate_speech(chunk, str(output_path), exaggeration)
            outputs.append(output_path)

        print(f"Done! Generated {len(outputs)} audio files")
        return outputs

    def combine_audio_files(self, audio_paths, output_path):
        """Combine nhiều audio files thành một."""
        all_audio = []

        for path in audio_paths:
            audio, sr = torchaudio.load(str(path))
            all_audio.append(audio)

            # Add small silence between chunks
            silence = torch.zeros(1, int(sr * 0.5))  # 0.5s silence
            all_audio.append(silence)

        combined = torch.cat(all_audio, dim=1)
        torchaudio.save(output_path, combined, sr)
        print(f"Combined audio saved to: {output_path}")
        return output_path


# Main usage
if __name__ == "__main__":
    # Initialize system
    system = VoiceLearningSystem(model_type="chatterbox", device="cuda")

    # Load reference voice
    system.load_reference_voice("reference_voice.wav")

    # Process a PDF document
    outputs = system.process_document(
        input_path="english_lesson.pdf",
        output_dir="outputs/lesson1/",
        exaggeration=0.5
    )

    # Combine all chunks into one file
    system.combine_audio_files(outputs, "outputs/lesson1_full.wav")
```

---

## 7. Tích Hợp Vào Ứng Dụng

### 7.1 REST API với FastAPI

```python
# api_server.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
import tempfile
import os
import torchaudio
from chatterbox.tts import ChatterboxTTS

app = FastAPI(title="Voice Cloning API")

# Global model
model = None
reference_audio = None

@app.on_event("startup")
async def load_model():
    global model
    model = ChatterboxTTS.from_pretrained(device="cuda")
    print("Model loaded!")

@app.post("/upload-reference")
async def upload_reference(file: UploadFile = File(...)):
    """Upload reference audio for voice cloning."""
    global reference_audio

    # Save uploaded file
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Load audio
    audio, sr = torchaudio.load(temp_path)
    if sr != model.sr:
        resampler = torchaudio.transforms.Resample(sr, model.sr)
        audio = resampler(audio)

    reference_audio = audio
    os.remove(temp_path)

    return {"status": "success", "message": "Reference audio uploaded"}

@app.post("/generate")
async def generate_speech(
    text: str = Form(...),
    exaggeration: float = Form(0.5)
):
    """Generate speech with cloned voice."""
    global model, reference_audio

    if reference_audio is None:
        return {"error": "Please upload reference audio first"}

    # Generate
    output = model.generate(
        text=text,
        audio_prompt=reference_audio,
        exaggeration=exaggeration
    )

    # Save to temp file
    temp_path = tempfile.mktemp(suffix=".wav")
    torchaudio.save(temp_path, output, model.sr)

    return FileResponse(temp_path, media_type="audio/wav", filename="output.wav")

@app.post("/process-pdf")
async def process_pdf(
    file: UploadFile = File(...),
    exaggeration: float = Form(0.5)
):
    """Process PDF and generate audio."""
    # Implementation similar to VoiceLearningSystem.process_document()
    pass

# Run: uvicorn api_server:app --host 0.0.0.0 --port 8000
```

### 7.2 Gradio Web Interface

```python
# gradio_app.py
import gradio as gr
import torchaudio
from chatterbox.tts import ChatterboxTTS

# Load model
model = ChatterboxTTS.from_pretrained(device="cuda")

def generate_speech(reference_audio, text, exaggeration):
    """Generate speech with voice cloning."""
    if reference_audio is None:
        return None, "Please upload reference audio!"

    # Load reference
    audio, sr = torchaudio.load(reference_audio)
    if sr != model.sr:
        resampler = torchaudio.transforms.Resample(sr, model.sr)
        audio = resampler(audio)

    # Generate
    output = model.generate(
        text=text,
        audio_prompt=audio,
        exaggeration=exaggeration
    )

    # Save
    output_path = "output.wav"
    torchaudio.save(output_path, output, model.sr)

    return output_path, "Generation successful!"

# Create interface
demo = gr.Interface(
    fn=generate_speech,
    inputs=[
        gr.Audio(label="Reference Audio", type="filepath"),
        gr.Textbox(label="Text to speak", lines=5),
        gr.Slider(0.0, 1.0, 0.5, label="Emotion Level")
    ],
    outputs=[
        gr.Audio(label="Generated Audio"),
        gr.Textbox(label="Status")
    ],
    title="Voice Cloning for English Learning",
    description="Upload a reference audio and enter text to generate speech with the cloned voice."
)

if __name__ == "__main__":
    demo.launch(share=True)

# Run: python gradio_app.py
```

### 7.3 Docker Deployment

```dockerfile
# Dockerfile
FROM nvidia/cuda:11.8-cudnn8-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  voice-cloning:
    build: .
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - ./models:/app/models
      - ./outputs:/app/outputs
```

---

## 8. Troubleshooting

### 8.1 Common Errors

#### CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```

**Solutions:**
```python
# 1. Reduce batch size
# 2. Use smaller chunks of text
# 3. Clear cache
import torch
torch.cuda.empty_cache()

# 4. Use CPU (slower)
model = ChatterboxTTS.from_pretrained(device="cpu")
```

#### Model Download Failed
```
Connection error when downloading model
```

**Solutions:**
```bash
# 1. Check internet connection
# 2. Use VPN if needed
# 3. Manual download from Hugging Face

# Set cache directory
export HF_HOME=/path/to/cache
export TRANSFORMERS_CACHE=/path/to/cache
```

#### Audio Quality Issues
```
Output audio sounds robotic or distorted
```

**Solutions:**
1. **Check reference audio quality**
   - Ensure clean audio without background noise
   - Use 5-30 seconds of clear speech

2. **Adjust parameters**
   ```python
   # Try different exaggeration levels
   output = model.generate(text=text, audio_prompt=ref, exaggeration=0.3)
   ```

3. **Preprocess reference audio**
   ```bash
   # Denoise and normalize
   ffmpeg -i input.mp3 -af "loudnorm,highpass=200,lowpass=3000" output.wav
   ```

#### Import Errors
```
ModuleNotFoundError: No module named 'xxx'
```

**Solutions:**
```bash
# 1. Ensure virtual environment is activated
source venv/bin/activate

# 2. Reinstall dependencies
pip install --upgrade -r requirements.txt

# 3. Check Python version
python --version  # Should be 3.10.x
```

### 8.2 Performance Optimization

#### Speed Up Inference
```python
# 1. Use half precision (FP16)
model = model.half()

# 2. Use torch.compile (PyTorch 2.0+)
model = torch.compile(model)

# 3. Batch processing
texts = ["text1", "text2", "text3"]
# Process multiple texts in one session
```

#### Reduce VRAM Usage
```python
# 1. Clear cache frequently
import gc
torch.cuda.empty_cache()
gc.collect()

# 2. Use gradient checkpointing
model.config.gradient_checkpointing = True

# 3. Process shorter texts
max_chars = 100  # Giới hạn độ dài text
```

### 8.3 Quality Tips

1. **Reference Audio**
   - Use clean, noise-free recordings
   - 10-30 seconds is optimal
   - Include variety of phonemes
   - Same speaker throughout

2. **Text Processing**
   - Split long texts into sentences
   - Avoid special characters
   - Add punctuation for natural pauses

3. **Generation Settings**
   - Start with default settings
   - Adjust exaggeration for emotion
   - Test different reference clips

---

## Appendix

### A. Requirements.txt

```txt
# Core
torch>=2.1.0
torchaudio>=2.1.0

# Voice Cloning
chatterbox-tts>=0.1.0
TTS>=0.22.0

# Audio Processing
pydub>=0.25.1
librosa>=0.10.0
soundfile>=0.12.1

# PDF Processing
PyPDF2>=3.0.0

# Web Framework
fastapi>=0.100.0
uvicorn>=0.23.0
gradio>=4.0.0

# Utilities
numpy>=1.24.0
scipy>=1.11.0
```

### B. Useful Links

| Resource | URL |
|----------|-----|
| Chatterbox GitHub | https://github.com/resemble-ai/chatterbox |
| GPT-SoVITS GitHub | https://github.com/RVC-Boss/GPT-SoVITS |
| Coqui TTS GitHub | https://github.com/coqui-ai/TTS |
| Hugging Face Models | https://huggingface.co/models?pipeline_tag=text-to-speech |
| CUDA Toolkit | https://developer.nvidia.com/cuda-toolkit |
| PyTorch | https://pytorch.org/get-started/locally/ |

### C. Hardware Recommendations

| Budget | GPU | VRAM | Performance |
|--------|-----|------|-------------|
| Budget | RTX 3060 | 12GB | Good |
| Mid-range | RTX 4070 | 12GB | Great |
| High-end | RTX 4090 | 24GB | Excellent |
| Cloud | A100/H100 | 40-80GB | Professional |

---

*Document Version: 1.0*
*Last Updated: January 2026*
