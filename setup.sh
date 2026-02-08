#!/bin/bash
set -e

echo "=== Tab Whisperer Voice Server Setup ==="

# Check dependencies
python3 --version || { echo "ERROR: Python 3.8+ required"; exit 1; }
cmake --version > /dev/null 2>&1 || {
    echo "ERROR: cmake is required to build whisper.cpp"
    echo "  macOS:  brew install cmake"
    echo "  Ubuntu: sudo apt install cmake"
    echo "  Fedora: sudo dnf install cmake"
    exit 1
}

# Clone and build whisper.cpp with CMake
if [ ! -f "whisper.cpp/build/bin/whisper-cli" ]; then
    if [ ! -d "whisper.cpp" ]; then
        echo "[1/4] Cloning whisper.cpp..."
        git clone https://github.com/ggml-org/whisper.cpp.git
    fi
    echo "[1/4] Building whisper.cpp (CMake)..."
    cd whisper.cpp
    cmake -B build
    cmake --build build -j --config Release
    cd ..
    echo "[1/4] Build complete: whisper.cpp/build/bin/whisper-cli"
else
    echo "[1/4] whisper.cpp already built"
fi

# Verify the binary works
if ! ./whisper.cpp/build/bin/whisper-cli --help > /dev/null 2>&1; then
    echo "ERROR: whisper-cli binary exists but failed to run"
    echo "Try rebuilding: cd whisper.cpp && rm -rf build && cmake -B build && cmake --build build -j --config Release"
    exit 1
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
python3 -c "
import sounddevice as sd
devices = sd.query_devices()
input_devices = [d for d in devices if d['max_input_channels'] > 0]
if input_devices:
    print(f'Found {len(input_devices)} input device(s):')
    for d in input_devices:
        print(f'  - {d[\"name\"]}')
else:
    print('WARNING: No input devices found!')
" || {
    echo "WARNING: Could not test microphone. Check permissions."
}

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Run the server:"
echo "  python3 server.py"
echo ""
echo "Test with: websocat ws://localhost:8765"