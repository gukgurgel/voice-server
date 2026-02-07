"""
Configuration for voice server.
Centralized settings for easy tuning during testing.
"""

# Audio capture
SAMPLE_RATE = 16000  # whisper.cpp expects 16kHz
CHANNELS = 1  # Mono
CHUNK_DURATION_MS = 500  # Process in 500ms chunks
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)

# VAD settings
VAD_THRESHOLD = 0.5  # Silero VAD threshold (0-1)
VAD_MIN_SPEECH_DURATION_MS = 250  # Minimum speech before considering it real
VAD_MIN_SILENCE_DURATION_MS = 700  # Silence duration to end utterance
VAD_SPEECH_PAD_MS = 300  # Padding before/after speech

# Whisper settings
WHISPER_MODEL = "base.en"  # Options: tiny.en, base.en, small.en
WHISPER_LANGUAGE = "en"
WHISPER_PATH = "./whisper.cpp/main"  # Path to compiled whisper.cpp binary
WHISPER_MODEL_PATH = "./models"  # Path to model files

# Wake word
WAKE_WORDS = ["hey tab", "tab whisperer"]  # Case-insensitive matches
WAKE_WORD_TIMEOUT_S = 5  # Timeout after wake if no command

# WebSocket
WS_HOST = "localhost"
WS_PORT = 8765

# Safety limits
MAX_RECORDING_DURATION_S = 30  # Maximum command length
MAX_BUFFER_SIZE_MB = 50  # Maximum memory for audio buffer