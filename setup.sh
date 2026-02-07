#!/bin/bash
set -e

echo "=== Tab Whisperer Voice Server Setup ==="

# Check Python version
python3 --version || { echo "Python 3.8+ required"; exit 1; }

# Clone and build whisper.cpp
if [ ! -d "whisper.cpp" ]; then
    echo "[1/4] Cloning whisper.cpp..."
    git clone https://github.com/ggml-org/whisper.cpp.git
    cd whisper.cpp
    make
    cd ..
else
    echo "[1/4] whisper.cpp already exists"
fi

# Download model
if [ ! -f "models/ggml-base.en.bin" ]; then
    echo "[2/4] Downloading Whisper base.en model..."
    mkdir -p models
    cd whisper.cpp
    bash ./models/download-ggml-model.sh base.en
    cp models/ggml-base.en.bin ../models/
    cd ..
else
    echo "[2/4] Model already downloaded"
fi

# Install Python dependencies
echo "[3/4] Installing Python dependencies..."
pip3 install -r requirements.txt

# Test microphone
echo "[4/4] Testing microphone access..."
python3 -c "import sounddevice as sd; print('Available devices:', sd.query_devices())" || {
    echo "WARNING: Could not access microphone. Check permissions."
}

echo ""
echo "=== Setup Complete ==="
echo "Run: python3 server.py"