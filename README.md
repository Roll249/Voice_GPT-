# Vietnamese Text-to-Speech Application

A comprehensive Vietnamese text-to-speech application that supports voice cloning and document processing. The application allows users to upload reference audio to clone voices and generate Vietnamese speech from text or PDF documents.

## Features

- **Voice Cloning**: Upload reference audio to clone voices
- **Multiple TTS Models**: Support for Chatterbox and XTTS-v2 models
- **Text-to-Speech**: Generate Vietnamese speech from text input
- **Document Processing**: Convert PDF documents to speech
- **Customizable Emotion**: Adjust emotion levels in generated speech
- **Web Interface**: User-friendly React frontend

## Architecture

- **Frontend**: React + TypeScript with Vite
- **Backend**: Python FastAPI server
- **TTS Models**: Chatterbox (primary) and XTTS-v2 (alternative)
- **OCR**: ddddocr for document processing
- **Audio Processing**: PyDub, Torchaudio

## Prerequisites

- Python 3.10+
- Node.js 18+ (for development)
- CUDA-compatible GPU (recommended for best performance)
- FFmpeg

## Installation

### Using the CLI Setup Tool

1. Navigate to the workspace directory:
```bash
cd /workspace
```

2. Run the setup tool:
```bash
# Make the setup script executable
chmod +x src/cli/setup.py

# Run the complete setup
python src/cli/setup.py --all
```

Or run specific setup steps:
```bash
# Check GPU availability
python src/cli/setup.py --check-gpu

# Install dependencies
python src/cli/setup.py --install-deps

# Setup backend components
python src/cli/setup.py --setup-backend
```

### Manual Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Install frontend dependencies:
```bash
cd src/frontend
npm install
```

## Running the Application

### Backend Server

1. Start the backend API server:
```bash
cd /workspace
uvicorn src.backend.api_server:app --reload --host 0.0.0.0 --port 8000
```

2. The API will be available at `http://localhost:8000`
3. API documentation available at `http://localhost:8000/docs`

### Frontend Development Server

1. In a separate terminal, navigate to the frontend directory:
```bash
cd /workspace/src/frontend
```

2. Start the development server:
```bash
npm run dev
```

3. The frontend will be available at `http://localhost:5173`

### Production Build

1. Build the frontend:
```bash
cd /workspace/src/frontend
npm run build
```

2. Serve the application using the backend server with static files serving capability.

## Usage

1. **Upload Reference Audio**: Upload a 5-30 second audio file of the voice you want to clone
2. **Select TTS Model**: Choose between Chatterbox (recommended) or XTTS-v2
3. **Adjust Emotion Level**: Control the expressiveness of the generated speech
4. **Generate Speech**: Enter text or upload a PDF document to generate speech

## API Endpoints

- `POST /upload-reference`: Upload reference audio for voice cloning
- `POST /generate`: Generate speech from text with cloned voice
- `POST /process-pdf`: Process PDF document and generate speech
- `GET /`: Health check endpoint

## Configuration

The application uses the following environment settings:

- `device`: Automatically detects CUDA availability, falls back to CPU if unavailable
- `model_type`: Default is 'chatterbox', can be switched to 'xtts'
- `exaggeration`: Controls emotion level (0.0 to 1.0)

## Directory Structure

```
/workspace/
├── docs/                   # Documentation files
├── src/
│   ├── backend/            # Python backend
│   ├── frontend/           # React frontend
│   ├── cli/                # CLI setup tools
│   └── utils/              # Utility scripts
├── models/                 # Downloaded models (created during setup)
├── uploads/                # Uploaded files (created during setup)
├── outputs/                # Generated audio files (created during setup)
├── references/             # Reference audio files (created during setup)
└── temp/                   # Temporary files (created during setup)
```

## Supported Languages

- Vietnamese (primary)
- English (secondary, through cross-language support)
- Other languages supported by XTTS-v2

## Troubleshooting

1. **CUDA Issues**: If CUDA is not available, the application will automatically fall back to CPU mode (slower)
2. **Memory Issues**: Large models may require significant RAM; consider reducing batch sizes
3. **Audio Quality**: Ensure reference audio is clear and noise-free for best results
4. **Model Downloads**: First run will download models (requires internet connection)

## License

This project uses MIT licensed components (Chatterbox, etc.) and is designed for both personal and commercial use.