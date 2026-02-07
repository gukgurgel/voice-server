"""
WebSocket server for voice commands.
Orchestrates audio capture, VAD, transcription, and state management.

Fixes from review:
- websockets v12 handler: (websocket) not (websocket, path)
- asyncio.get_running_loop() instead of deprecated get_event_loop()
- Correct VAD_CHUNK_SAMPLES for sounddevice blocksize
- Single-client guard with proper disconnect
- Graceful WebSocket shutdown
"""

import asyncio
import websockets
import json
import sounddevice as sd
import numpy as np
from typing import Optional
import signal
import sys

from config import (
    SAMPLE_RATE, CHANNELS, VAD_CHUNK_SAMPLES,
    WHISPER_MODEL, WAKE_WORDS, WS_HOST, WS_PORT
)
from vad_detector import VADDetector
from transcriber import WhisperTranscriber
from state_machine import StateMachine, State


class VoiceServer:
    def __init__(self):
        self.vad = VADDetector()
        self.transcriber = WhisperTranscriber()
        self.state_machine: Optional[StateMachine] = None

        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.audio_stream: Optional[sd.InputStream] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.running = False
        self._shutdown_event: Optional[asyncio.Event] = None

        print(f"[Server] Initialized")

    async def send_message(self, msg_type: str, **kwargs):
        """Send JSON message to connected client."""
        if not self.websocket:
            return

        message = {"type": msg_type, **kwargs}
        try:
            await self.websocket.send(json.dumps(message))
            print(f"[Server] Sent: {message}")
        except websockets.exceptions.ConnectionClosed:
            print(f"[Server] Client disconnected during send")
            self.websocket = None
        except Exception as e:
            print(f"[Server] Send error: {e}")

    # --- State machine callbacks ---

    async def on_wake(self):
        await self.send_message("wake")

    async def on_listening(self):
        await self.send_message("listening")

    async def on_command(self, text: str):
        await self.send_message("command", text=text)

    async def on_error(self, error: str):
        await self.send_message("error", message=error)

    # --- Audio pipeline ---

    def audio_callback(self, indata, frames, time_info, status):
        """
        Audio input callback from sounddevice.
        Called in a separate thread — must not block or do async directly.
        """
        if status:
            print(f"[Audio] Status: {status}")

        if not self.running or self.loop is None:
            return

        # Copy audio data (indata is a temporary buffer view)
        audio_chunk = indata[:, 0].copy()  # Extract mono channel

        # Schedule async processing in the event loop (thread-safe)
        asyncio.run_coroutine_threadsafe(
            self.process_audio_chunk(audio_chunk),
            self.loop
        )

    async def process_audio_chunk(self, audio_chunk: np.ndarray):
        """Process incoming audio chunk through VAD → transcription pipeline."""
        try:
            # Run VAD on the chunk
            is_speech, utterance = self.vad.process_chunk(audio_chunk)

            # If a complete utterance was detected, transcribe it
            if utterance is not None:
                text = await self.transcriber.transcribe(utterance)

                if text:
                    await self.state_machine.process_transcription(text)

        except Exception as e:
            print(f"[Server] Processing error: {e}")
            await self.on_error(str(e))

    # --- WebSocket handling ---

    async def handle_client_message(self, message: str):
        """Handle incoming WebSocket messages from extension."""
        try:
            data = json.loads(message)
            msg_type = data.get("type")

            print(f"[Server] Received: {data}")

            if msg_type == "start_listening":
                await self.state_machine.manual_trigger()

            elif msg_type == "stop_listening":
                await self.state_machine.stop()

            elif msg_type == "config":
                print(f"[Server] Config update: {data}")

            elif msg_type == "ack":
                print(f"[Server] Command acknowledged: {data.get('result')}")

            else:
                print(f"[Server] Unknown message type: {msg_type}")

        except json.JSONDecodeError as e:
            print(f"[Server] Invalid JSON: {e}")
        except Exception as e:
            print(f"[Server] Message handling error: {e}")

    async def websocket_handler(self, websocket):
        """
        Handle WebSocket connection from extension.

        Note: websockets v12+ handler takes (websocket) only, not (websocket, path).
        Only one client at a time — new connections replace the old one.
        """
        # If another client is already connected, close it cleanly
        if self.websocket is not None:
            print(f"[Server] Replacing existing client connection")
            try:
                await self.websocket.close(1000, "Replaced by new connection")
            except Exception:
                pass

        self.websocket = websocket
        print(f"[Server] Client connected from {websocket.remote_address}")

        # Send initial status so extension knows server capabilities
        await self.send_message(
            "status",
            vad=True,
            model=WHISPER_MODEL,
            wake_word=" or ".join(WAKE_WORDS)
        )

        try:
            async for message in websocket:
                await self.handle_client_message(message)
        except websockets.exceptions.ConnectionClosed:
            print(f"[Server] Client disconnected")
        finally:
            if self.websocket is websocket:
                self.websocket = None

    # --- Audio stream management ---

    def start_audio_stream(self):
        """Start audio capture from microphone."""
        try:
            # List available input devices for debugging
            devices = sd.query_devices()
            print(f"[Audio] Available input devices:")
            for i, dev in enumerate(devices):
                if dev['max_input_channels'] > 0:
                    marker = " (default)" if i == sd.default.device[0] else ""
                    print(f"  [{i}] {dev['name']}{marker}")

            # Start stream with VAD-compatible chunk size
            self.audio_stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype='float32',
                blocksize=VAD_CHUNK_SAMPLES,  # MUST match Silero's expected chunk size
                callback=self.audio_callback
            )
            self.audio_stream.start()
            print(f"[Audio] Stream started (rate={SAMPLE_RATE}Hz, "
                  f"blocksize={VAD_CHUNK_SAMPLES} samples)")

        except Exception as e:
            print(f"[Audio] ERROR: Failed to start audio stream: {e}")
            print(f"[Audio] Make sure you have a working microphone")
            sys.exit(1)

    def stop_audio_stream(self):
        """Stop audio capture."""
        if self.audio_stream:
            try:
                self.audio_stream.stop()
                self.audio_stream.close()
            except Exception as e:
                print(f"[Audio] Error stopping stream: {e}")
            self.audio_stream = None
            print(f"[Audio] Stream stopped")

    # --- Main server lifecycle ---

    async def run(self):
        """Main server loop."""
        self.running = True
        self.loop = asyncio.get_running_loop()  # Not get_event_loop() — deprecated in 3.10+
        self._shutdown_event = asyncio.Event()

        # Initialize state machine with callbacks
        self.state_machine = StateMachine(
            on_wake=self.on_wake,
            on_listening=self.on_listening,
            on_command=self.on_command,
            on_error=self.on_error
        )

        # Start audio capture
        self.start_audio_stream()

        # Start WebSocket server
        print(f"[Server] Starting WebSocket on ws://{WS_HOST}:{WS_PORT}")

        async with websockets.serve(self.websocket_handler, WS_HOST, WS_PORT):
            print(f"[Server] Ready! Wake words: {WAKE_WORDS}")
            print(f"[Server] Waiting for extension to connect...")

            # Run until shutdown is signaled
            await self._shutdown_event.wait()

    async def shutdown(self):
        """Clean shutdown of all components."""
        print(f"\n[Server] Shutting down...")
        self.running = False

        # Close WebSocket client
        if self.websocket:
            try:
                await self.websocket.close(1001, "Server shutting down")
            except Exception:
                pass
            self.websocket = None

        # Stop audio
        self.stop_audio_stream()

        # Signal main loop to exit
        if self._shutdown_event:
            self._shutdown_event.set()


async def main():
    """Entry point."""
    server = VoiceServer()

    # Handle Ctrl+C gracefully
    loop = asyncio.get_running_loop()

    def signal_handler():
        asyncio.ensure_future(server.shutdown())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    try:
        await server.run()
    except Exception as e:
        print(f"[Server] Fatal error: {e}")
        await server.shutdown()


if __name__ == "__main__":
    asyncio.run(main())