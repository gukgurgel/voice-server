"""
Whisper transcription using whisper.cpp for maximum performance.
Uses subprocess to call the compiled binary.
"""

import subprocess
import tempfile
import wave
import numpy as np
from pathlib import Path
from typing import Optional
import asyncio
from config import *

class WhisperTranscriber:
    def __init__(self):
        """Initialize whisper.cpp transcriber."""
        self.whisper_path = Path(WHISPER_PATH)
        self.model_path = Path(WHISPER_MODEL_PATH) / f"ggml-{WHISPER_MODEL}.bin"
        
        # Validate paths
        if not self.whisper_path.exists():
            raise FileNotFoundError(
                f"whisper.cpp binary not found at {self.whisper_path}\n"
                f"Build it: cd whisper.cpp && make"
            )
        
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}\n"
                f"Download: ./whisper.cpp/models/download-ggml-model.sh {WHISPER_MODEL}"
            )
        
        print(f"[Whisper] Initialized with model {WHISPER_MODEL}")
        print(f"[Whisper] Binary: {self.whisper_path}")
        print(f"[Whisper] Model: {self.model_path}")
    
    async def transcribe(self, audio: np.ndarray) -> Optional[str]:
        """
        Transcribe audio using whisper.cpp.
        
        Args:
            audio: numpy array of audio samples (16kHz, mono, float32)
        
        Returns:
            Transcribed text or None if transcription failed
        """
        # Safety check: audio length
        duration_s = len(audio) / SAMPLE_RATE
        if duration_s > MAX_RECORDING_DURATION_S:
            print(f"[Whisper] WARNING: Audio too long ({duration_s:.1f}s), truncating")
            audio = audio[:MAX_RECORDING_DURATION_S * SAMPLE_RATE]
        
        if duration_s < 0.1:
            print(f"[Whisper] Audio too short ({duration_s:.2f}s), skipping")
            return None
        
        # Write to temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            
            try:
                # Convert float32 to int16 for WAV
                audio_int16 = (audio * 32767).astype(np.int16)
                
                with wave.open(tmp_path, 'wb') as wav_file:
                    wav_file.setnchannels(CHANNELS)
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(SAMPLE_RATE)
                    wav_file.writeframes(audio_int16.tobytes())
                
                # Run whisper.cpp in subprocess
                result = await self._run_whisper(tmp_path)
                return result
                
            finally:
                # Clean up temp file
                Path(tmp_path).unlink(missing_ok=True)
    
    async def _run_whisper(self, audio_path: str) -> Optional[str]:
        """Run whisper.cpp binary and parse output."""
        cmd = [
            str(self.whisper_path),
            '-m', str(self.model_path),
            '-f', audio_path,
            '-l', WHISPER_LANGUAGE,
            '-t', '4',  # 4 threads
            '--no-timestamps',  # We don't need timestamps
            '--output-txt',  # Output as text
            '--print-colors',  # Disable color codes
        ]
        
        try:
            # Run in thread pool to avoid blocking event loop
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=10.0  # 10s timeout
            )
            
            # Parse output
            output = stdout.decode('utf-8').strip()
            
            # whisper.cpp outputs format like:
            # [00:00:00.000 --> 00:00:02.000]   group my tabs by topic
            # We need to extract just the text
            lines = output.split('\n')
            text_lines = []
            
            for line in lines:
                # Skip empty lines and metadata
                line = line.strip()
                if not line or line.startswith('['):
                    continue
                
                # Remove timestamp prefix if present
                if ']' in line:
                    line = line.split(']', 1)[1].strip()
                
                text_lines.append(line)
            
            text = ' '.join(text_lines).strip()
            
            if text:
                print(f"[Whisper] Transcribed: '{text}'")
                return text
            else:
                print(f"[Whisper] No transcription output")
                return None
                
        except asyncio.TimeoutError:
            print(f"[Whisper] ERROR: Transcription timeout")
            return None
        except Exception as e:
            print(f"[Whisper] ERROR: {e}")
            return None