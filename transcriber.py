"""
Whisper transcription using whisper.cpp for maximum performance.
Uses subprocess to call the compiled binary.

Key fixes from review:
- Pipe PCM via temp WAV (stdin pipe not supported by whisper-cli for WAV header)
- Use -otxt + --output-file for clean text output (no stdout parsing needed)
- Removed --print-colors (that ENABLES color codes, not disables)
- Removed --output-txt (writes file next to input, orphans temp files)
- Dynamic thread count from config
"""

import subprocess
import tempfile
import wave
import os
import numpy as np
from pathlib import Path
from typing import Optional
import asyncio
from config import (
    SAMPLE_RATE, CHANNELS, WHISPER_PATH, WHISPER_MODEL_PATH,
    WHISPER_PASSIVE_MODEL, WHISPER_ACTIVE_MODEL, WHISPER_LANGUAGE,
    WHISPER_THREADS, MAX_RECORDING_DURATION_S
)


class WhisperTranscriber:
    def __init__(self):
        """Initialize whisper.cpp transcriber with passive + active models."""
        self.whisper_path = Path(WHISPER_PATH)
        self.passive_model_path = Path(WHISPER_MODEL_PATH) / f"ggml-{WHISPER_PASSIVE_MODEL}.bin"
        self.active_model_path = Path(WHISPER_MODEL_PATH) / f"ggml-{WHISPER_ACTIVE_MODEL}.bin"

        # Validate binary at startup — fail fast
        if not self.whisper_path.exists():
            raise FileNotFoundError(
                f"whisper.cpp binary not found at {self.whisper_path}\n"
                f"Build it:\n"
                f"  cd whisper.cpp\n"
                f"  cmake -B build\n"
                f"  cmake --build build -j --config Release"
            )

        # Validate both models at startup
        for label, model_name, model_path in [
            ("Passive", WHISPER_PASSIVE_MODEL, self.passive_model_path),
            ("Active", WHISPER_ACTIVE_MODEL, self.active_model_path),
        ]:
            if not model_path.exists():
                raise FileNotFoundError(
                    f"{label} model not found at {model_path}\n"
                    f"Download:\n"
                    f"  cd whisper.cpp\n"
                    f"  bash ./models/download-ggml-model.sh {model_name}"
                )

        print(f"[Whisper] Passive model: {WHISPER_PASSIVE_MODEL} ({self.passive_model_path})")
        print(f"[Whisper] Active model:  {WHISPER_ACTIVE_MODEL} ({self.active_model_path})")
        print(f"[Whisper] Binary: {self.whisper_path}")
        print(f"[Whisper] Threads: {WHISPER_THREADS}")

    async def transcribe(self, audio: np.ndarray, active: bool = False) -> Optional[str]:
        """
        Transcribe audio using whisper.cpp.

        Args:
            audio: numpy array of audio samples (16kHz, mono, float32, range [-1, 1])
            active: if True, use the high-quality active model (for commands);
                    if False, use the lightweight passive model (for wake word)

        Returns:
            Transcribed text or None if transcription failed/empty
        """
        duration_s = len(audio) / SAMPLE_RATE

        # Safety: reject too-short audio (just noise clicks)
        if duration_s < 0.3:
            print(f"[Whisper] Audio too short ({duration_s:.2f}s), skipping")
            return None

        # Safety: truncate excessively long audio
        if duration_s > MAX_RECORDING_DURATION_S:
            print(f"[Whisper] WARNING: Audio too long ({duration_s:.1f}s), truncating to {MAX_RECORDING_DURATION_S}s")
            audio = audio[:MAX_RECORDING_DURATION_S * SAMPLE_RATE]
            duration_s = MAX_RECORDING_DURATION_S

        model_path = self.active_model_path if active else self.passive_model_path
        model_label = "active/turbo" if active else "passive/base"
        print(f"[Whisper] Transcribing {duration_s:.1f}s of audio ({model_label})...")

        # We need two temp files:
        #   1. WAV input (whisper-cli needs a file, no stdin WAV support)
        #   2. TXT output (we use --output-file to get clean text, no stdout parsing)
        tmp_wav = None
        tmp_out = None

        try:
            # Write audio to temp WAV
            tmp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            tmp_wav_path = tmp_wav.name
            tmp_wav.close()  # Close so whisper-cli can read it

            audio_int16 = (audio * 32767).astype(np.int16)
            with wave.open(tmp_wav_path, 'wb') as wav_file:
                wav_file.setnchannels(CHANNELS)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(SAMPLE_RATE)
                wav_file.writeframes(audio_int16.tobytes())

            # Prepare output path (whisper-cli appends .txt to --output-file value)
            tmp_out = tempfile.NamedTemporaryFile(suffix='', delete=False)
            tmp_out_base = tmp_out.name
            tmp_out.close()
            tmp_out_txt = tmp_out_base + ".txt"

            # Run whisper-cli
            text = await self._run_whisper(tmp_wav_path, tmp_out_base, model_path)
            return text

        except Exception as e:
            print(f"[Whisper] ERROR: {e}")
            return None

        finally:
            # Clean up all temp files
            for path in [tmp_wav_path if tmp_wav else None,
                         tmp_out_base if tmp_out else None,
                         tmp_out_base + ".txt" if tmp_out else None]:
                if path:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass

    async def _run_whisper(self, audio_path: str, output_base: str, model_path: Path) -> Optional[str]:
        """
        Run whisper.cpp binary and return transcribed text.

        Strategy: Use -otxt --output-file to write clean text to a file.
        This avoids all stdout parsing issues (timestamps, color codes, etc).
        """
        cmd = [
            str(self.whisper_path),
            '-m', str(model_path),
            '-f', audio_path,
            '-l', WHISPER_LANGUAGE,
            '-t', str(WHISPER_THREADS),
            '-otxt',                         # Output as .txt file
            '--output-file', output_base,    # Write to this path (+ .txt appended)
            '--no-prints',                   # Suppress progress/debug to stderr
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=15.0  # generous timeout for longer audio
            )

            if process.returncode != 0:
                print(f"[Whisper] Process exited with code {process.returncode}")
                if stderr:
                    print(f"[Whisper] stderr: {stderr.decode('utf-8', errors='replace')[:500]}")
                return None

            # Read the clean text output file
            output_txt_path = output_base + ".txt"
            if not os.path.exists(output_txt_path):
                print(f"[Whisper] Output file not created: {output_txt_path}")
                # Fallback: try parsing stdout
                return self._parse_stdout(stdout.decode('utf-8', errors='replace'))

            with open(output_txt_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()

            if text:
                print(f"[Whisper] Transcribed: '{text}'")
                return text
            else:
                print(f"[Whisper] Empty transcription")
                return None

        except asyncio.TimeoutError:
            print(f"[Whisper] ERROR: Transcription timeout (>15s)")
            return None
        except Exception as e:
            print(f"[Whisper] ERROR: {e}")
            return None

    @staticmethod
    def _parse_stdout(output: str) -> Optional[str]:
        """
        Fallback: parse whisper-cli stdout if file output failed.
        Handles both timestamped and plain output formats.
        """
        lines = output.strip().split('\n')
        text_parts = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Strip timestamp prefix like "[00:00:00.000 --> 00:00:02.000]"
            if line.startswith('[') and ']' in line:
                _, _, remainder = line.partition(']')
                line = remainder.strip()

            if line:
                text_parts.append(line)

        text = ' '.join(text_parts).strip()
        return text if text else None