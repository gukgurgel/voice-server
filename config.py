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
VAD_MIN_SPEECH_DURATION_MS = 250  # Minimum speech before considering it real
VAD_MIN_SILENCE_DURATION_MS = 700  # Silence duration to end utterance
VAD_SPEECH_PAD_MS = 300  # Padding before/after speech

# Whisper settings
WHISPER_MODEL = "base.en"  # Options: tiny.en, base.en, small.en
WHISPER_LANGUAGE = "en"
WHISPER_PATH = "./whisper.cpp/build/bin/whisper-cli"  # CMake build output
WHISPER_MODEL_PATH = "./models"  # Path to model files
WHISPER_THREADS = os.cpu_count() or 4  # Match available CPU cores

# Wake word
WAKE_WORDS = ["hey fox"]  # Case-insensitive matches
WAKE_WORD_TIMEOUT_S = 10  # Timeout after wake if no command (was 5 — too short)

# WebSocket
WS_HOST = "localhost"
WS_PORT = 8765

# Safety limits
MAX_RECORDING_DURATION_S = 30  # Maximum command length
MAX_AUDIO_BUFFER_FRAMES = int(MAX_RECORDING_DURATION_S * SAMPLE_RATE / VAD_CHUNK_SAMPLES)