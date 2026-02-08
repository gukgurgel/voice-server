"""
Configuration for voice server.
Centralized settings for easy tuning during testing.
"""

import os

# Audio capture
SAMPLE_RATE = 16000  # whisper.cpp expects 16kHz
CHANNELS = 1  # Mono

# VAD chunk size: Silero VAD streaming API requires EXACTLY one of:
#   512, 1024, or 1536 samples at 16kHz (32ms, 64ms, 96ms)
# 512 = lowest latency (~32ms), Silero inference is ~0.1ms so this is fine
VAD_CHUNK_SAMPLES = 512
VAD_CHUNK_DURATION_MS = VAD_CHUNK_SAMPLES * 1000 // SAMPLE_RATE  # 32ms

# VAD settings
VAD_THRESHOLD = 0.5  # Silero VAD threshold (0-1)
VAD_MIN_SPEECH_DURATION_MS = 100  # Minimum speech before considering it real
VAD_MIN_SILENCE_DURATION_MS = 400  # Default silence duration (unused directly — see passive/active)
VAD_PASSIVE_SILENCE_MS = 300  # Shorter silence for wake word (brief phrase, don't wait long)
VAD_ACTIVE_SILENCE_MS = 700  # Longer silence for commands (natural pauses in speech)
VAD_SPEECH_PAD_MS = 300  # Padding before/after speech

# Whisper settings
# Passive model: lightweight, used for wake word detection (runs constantly)
WHISPER_PASSIVE_MODEL = "base.en"     # Fast & cheap — just needs to catch "hey fox"
# Active model: high-quality, used for command transcription (runs once per command)
WHISPER_ACTIVE_MODEL = "large-v3-turbo"  # Best accuracy for actual commands
WHISPER_LANGUAGE = "en"
WHISPER_PATH = "./whisper.cpp/build/bin/whisper-cli"  # CMake build output
WHISPER_MODEL_PATH = "./models"  # Path to model files
WHISPER_THREADS = os.cpu_count() or 4  # Match available CPU cores

# Wake word
WAKE_WORDS = ["hey fox"]  # Case-insensitive matches
WAKE_WORD_TIMEOUT_S = 20  # Timeout after wake if no command (was 5 — too short)

# WebSocket
WS_HOST = "localhost"
WS_PORT = 8765

# Safety limits
MAX_RECORDING_DURATION_S = 30  # Maximum command length
MAX_AUDIO_BUFFER_FRAMES = int(MAX_RECORDING_DURATION_S * SAMPLE_RATE /
                              VAD_CHUNK_SAMPLES)

