"""
Voice Activity Detection using Silero VAD.
Lightweight, CPU-only, highly accurate speech detection.

CRITICAL: Silero VAD streaming requires chunks of exactly 512, 1024, or 1536
samples at 16kHz. The audio callback blocksize MUST match VAD_CHUNK_SAMPLES.
"""

import torch
import numpy as np
from typing import Optional
from config import (
    SAMPLE_RATE, VAD_THRESHOLD, VAD_MIN_SPEECH_DURATION_MS,
    VAD_PASSIVE_SILENCE_MS, VAD_ACTIVE_SILENCE_MS, VAD_CHUNK_DURATION_MS,
    MAX_AUDIO_BUFFER_FRAMES
)


class VADDetector:
    def __init__(self):
        """Initialize Silero VAD model."""
        # Load pre-trained Silero VAD
        self.model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )

        self.model.eval()

        # State tracking
        self.speech_started = False
        self.speech_frames: list[np.ndarray] = []
        self.silence_frames = 0

        # Pre-compute frame counts from ms durations
        self.min_speech_frames = max(1, int(
            VAD_MIN_SPEECH_DURATION_MS / VAD_CHUNK_DURATION_MS
        ))

        # Dynamic silence thresholds for passive (wake word) vs active (command) modes
        self._passive_silence_frames = max(1, int(
            VAD_PASSIVE_SILENCE_MS / VAD_CHUNK_DURATION_MS
        ))
        self._active_silence_frames = max(1, int(
            VAD_ACTIVE_SILENCE_MS / VAD_CHUNK_DURATION_MS
        ))

        # Start in passive mode (shorter silence timeout for wake word detection)
        self.min_silence_frames = self._passive_silence_frames

        print(f"[VAD] Initialized (threshold={VAD_THRESHOLD}, "
              f"chunk={VAD_CHUNK_DURATION_MS}ms, "
              f"min_speech={self.min_speech_frames} frames, "
              f"passive_silence={self._passive_silence_frames} frames, "
              f"active_silence={self._active_silence_frames} frames)")

    def set_mode(self, mode: str):
        """
        Switch VAD silence timeout between passive and active modes.

        Args:
            mode: "passive" (short silence for wake word) or "active" (long silence for commands)
        """
        if mode == "passive":
            self.min_silence_frames = self._passive_silence_frames
        elif mode == "active":
            self.min_silence_frames = self._active_silence_frames
        else:
            raise ValueError(f"Unknown VAD mode: {mode}")
        print(f"[VAD] Mode → {mode} (silence={self.min_silence_frames} frames)")

    def process_chunk(self, audio_chunk: np.ndarray) -> tuple[bool, Optional[np.ndarray]]:
        """
        Process a single audio chunk and return (is_speech, completed_utterance).

        Args:
            audio_chunk: numpy array, must be exactly VAD_CHUNK_SAMPLES long

        Returns:
            is_speech: True if current chunk contains speech
            completed_utterance: Full audio array if utterance ended, else None
        """
        # Convert to torch tensor (float32, range [-1, 1])
        audio_tensor = torch.from_numpy(audio_chunk).float()

        # Get VAD prediction
        with torch.no_grad():
            speech_prob = self.model(audio_tensor, SAMPLE_RATE).item()

        is_speech = speech_prob > VAD_THRESHOLD

        if is_speech:
            self.speech_frames.append(audio_chunk)
            self.silence_frames = 0

            if not self.speech_started and len(self.speech_frames) >= self.min_speech_frames:
                self.speech_started = True
                print(f"[VAD] Speech started (prob={speech_prob:.2f})")

            # Safety: prevent unbounded buffer growth
            if len(self.speech_frames) >= MAX_AUDIO_BUFFER_FRAMES:
                print(f"[VAD] Buffer limit reached, forcing utterance end")
                utterance = np.concatenate(self.speech_frames)
                self.reset()
                return is_speech, utterance

        else:
            # Silence
            if self.speech_started:
                self.silence_frames += 1
                self.speech_frames.append(audio_chunk)  # Include trailing silence

                # End of utterance?
                if self.silence_frames >= self.min_silence_frames:
                    print(f"[VAD] Speech ended ({len(self.speech_frames)} frames)")
                    utterance = np.concatenate(self.speech_frames)
                    self.reset()
                    return is_speech, utterance
            else:
                # Pre-speech silence — discard accumulated non-speech frames
                # but keep a small lookback buffer for speech onset
                if len(self.speech_frames) > self.min_speech_frames:
                    self.speech_frames = []

        return is_speech, None

    def force_end_utterance(self) -> Optional[np.ndarray]:
        """Force end current utterance (e.g., on timeout). Returns accumulated audio."""
        if len(self.speech_frames) > 0:
            print(f"[VAD] Force ending utterance ({len(self.speech_frames)} frames)")
            utterance = np.concatenate(self.speech_frames)
            self.reset()
            return utterance
        return None

    def reset(self):
        """Reset VAD state including Silero's internal LSTM state."""
        self.speech_started = False
        self.speech_frames = []
        self.silence_frames = 0
        # CRITICAL: Reset Silero's internal hidden states (h, c tensors)
        # Without this, LSTM state from previous utterance leaks into next detection
        self.model.reset_states()