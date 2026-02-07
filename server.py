"""
WebSocket server for voice commands.
Orchestrates audio capture, VAD, transcription, and state management.
"""

import asyncio
import websockets
import json
import sounddevice as sd
import numpy as np
from typing import Optional
import signal
import sys

from config import *
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
        self.running = False
        
        print(f"[Server] Initialized")
    
    async def send_message(self, msg_type: str, **kwargs):
        """Send JSON message to connected client."""
        if not self.websocket:
            return
        
        message = {"type": msg_type, **kwargs}
        try:
            await self.websocket.send(json.dumps(message))
            print(f"[Server] Sent: {message}")
        except Exception as e:
            print(f"[Server] Send error: {e}")
    
    async def on_wake(self):
        """Callback: wake word detected."""
        await self.send_message("wake")
    
    async def on_listening(self):
        """Callback: actively listening for command."""
        await self.send_message("listening")
    
    async def on_command(self, text: str):
        """Callback: command transcribed."""
        await self.send_message("command", text=text)
    
    async def on_error(self, error: str):
        """Callback: error occurred."""
        await self.send_message("error", message=error)
    
    def audio_callback(self, indata, frames, time_info, status):
        """
        Audio input callback from sounddevice.
        Called in separate thread - must be thread-safe.
        """
        if status:
            print(f"[Audio] Status: {status}")
        
        # Copy audio data (indata is a view)
        audio_chunk = indata[:, 0].copy()  # Extract mono channel
        
        # Schedule processing in event loop
        asyncio.run_coroutine_threadsafe(
            self.process_audio_chunk(audio_chunk),
            self.loop
        )
    
    async def process_audio_chunk(self, audio_chunk: np.ndarray):
        """Process incoming audio chunk through VAD and transcription pipeline."""
        try:
            # Run VAD
            is_speech, utterance = self.vad.process_chunk(audio_chunk)
            
            # If utterance completed, transcribe it
            if utterance is not None:
                text = await self.transcriber.transcribe(utterance)
                
                if text:
                    # Process through state machine
                    await self.state_machine.process_transcription(text)
                    
        except Exception as e:
            print(f"[Server] Process error: {e}")
            await self.on_error(str(e))
    
    async def handle_client_message(self, message: str):
        """Handle incoming WebSocket messages from extension."""
        try:
            data = json.loads(message)
            msg_type = data.get("type")
            
            print(f"[Server] Received: {data}")
            
            if msg_type == "start_listening":
                # Manual trigger - go straight to active mode
                await self.state_machine.manual_trigger()
            
            elif msg_type == "stop_listening":
                # Cancel current session
                await self.state_machine.stop()
            
            elif msg_type == "config":
                # Configuration update (could add wake word customization)
                print(f"[Server] Config update: {data}")
            
            elif msg_type == "ack":
                # Acknowledgment from extension (for future TTS)
                print(f"[Server] Command acknowledged: {data.get('result')}")
            
            else:
                print(f"[Server] Unknown message type: {msg_type}")
                
        except json.JSONDecodeError as e:
            print(f"[Server] Invalid JSON: {e}")
        except Exception as e:
            print(f"[Server] Message handling error: {e}")
    
    async def websocket_handler(self, websocket, path):
        """Handle WebSocket connection from extension."""
        self.websocket = websocket
        print(f"[Server] Client connected from {websocket.remote_address}")
        
        # Send initial status
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
            self.websocket = None
    
    def start_audio_stream(self):
        """Start audio capture from microphone."""
        try:
            # List available devices
            devices = sd.query_devices()
            print(f"[Audio] Available devices:")
            for i, dev in enumerate(devices):
                if dev['max_input_channels'] > 0:
                    print(f"  [{i}] {dev['name']}")
            
            # Start stream
            self.audio_stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype='float32',
                blocksize=CHUNK_SIZE,
                callback=self.audio_callback
            )
            self.audio_stream.start()
            print(f"[Audio] Stream started (rate={SAMPLE_RATE}Hz, chunk={CHUNK_DURATION_MS}ms)")
            
        except Exception as e:
            print(f"[Audio] ERROR: Failed to start audio stream: {e}")
            print(f"[Audio] Make sure you have a working microphone")
            sys.exit(1)
    
    def stop_audio_stream(self):
        """Stop audio capture."""
        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
            print(f"[Audio] Stream stopped")
    
    async def run(self):
        """Main server loop."""
        self.running = True
        self.loop = asyncio.get_event_loop()
        
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
        print(f"[Server] Starting WebSocket server on {WS_HOST}:{WS_PORT}")
        async with websockets.serve(self.websocket_handler, WS_HOST, WS_PORT):
            print(f"[Server] Ready! Listening for wake words: {WAKE_WORDS}")
            print(f"[Server] Waiting for extension to connect...")
            
            # Run forever
            await asyncio.Future()  # Run until cancelled
    
    def shutdown(self):
        """Clean shutdown."""
        print(f"\n[Server] Shutting down...")
        self.running = False
        self.stop_audio_stream()

async def main():
    """Entry point."""
    server = VoiceServer()
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        server.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        await server.run()
    except KeyboardInterrupt:
        server.shutdown()

if __name__ == "__main__":
    asyncio.run(main())