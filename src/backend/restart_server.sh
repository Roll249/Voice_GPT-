#!/bin/bash
# Kill existing server
pkill -f "simple_api_server"
sleep 2
# Start new server
cd /home/khang/khang_lab/Voice_GPT-
python -c "
import sys
sys.path.insert(0, '/home/khang/khang_lab/Voice_GPT-')
from src.backend.simple_api_server import app
import uvicorn
print('Starting Vietnamese TTS API server...')
uvicorn.run(app, host='0.0.0.0', port=8000, reload=True)
" &
echo "Server restarted"
