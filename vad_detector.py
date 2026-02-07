"""
Voice Activity Detection using Silero VAD.
Lightweight, CPU-only, highly accurate speech detection.
"""

import torch
import numpy as np
from typing import Optional
from config import *

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
        
        (self.get_speech_timestamps,
         self.save_audio,
         self.read_audio,
         self.VADIterator,
         self.collect_chunks) = utils
        
        self.model.eval()
        
        # State tracking
        self.speech_started = False
        self.speech_frames = []
        self.silence_frames = 0
        
        self.min_speech_frames = int(
            VAD_MIN_SPEECH_DURATION_MS / CHUNK_DURATION_MS
        )
        self.min_silence_frames = int(
            VAD_MIN_SILENCE_DURATION_MS / CHUNK_DURATION_MS
        )
        
        print(f"[VAD] Initialized (threshold={VAD_THRESHOLD})")
    
    def process_chunk(self, audio_chunk: np.ndarray) -> tuple[bool, Optional[np.ndarray]]:
        """
        Process an audio chunk and return (is_speech, completed_utterance).
        
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
        
        # State machine for utterance detection
        if is_speech:
            self.speech_frames.append(audio_chunk)
            self.silence_frames = 0
            
            if not self.speech_started:
                # Speech just started
                if len(self.speech_frames) >= self.min_speech_frames:
                    self.speech_started = True
                    print(f"[VAD] Speech started (prob={speech_prob:.2f})")
        else:
            # Silence detected
            if self.speech_started:
                self.silence_frames += 1
                self.speech_frames.append(audio_chunk)  # Include silence for natural endings
                
                # End of utterance?
                if self.silence_frames >= self.min_silence_frames:
                    print(f"[VAD] Speech ended ({len(self.speech_frames)} frames)")
                    utterance = np.concatenate(self.speech_frames)
                    self.reset()
                    return is_speech, utterance
            else:
                # Still in initial silence, discard old frames
                if len(self.speech_frames) > 0:
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
        """Reset VAD state."""
        self.speech_started = False
        self.speech_frames = []
        self.silence_frames = 0