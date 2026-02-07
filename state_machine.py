"""
State machine for wake word detection and command processing.
Two states: PASSIVE (listening for wake word) and ACTIVE (recording command).
"""

from enum import Enum
from typing import Optional, Callable
import asyncio
from config import WAKE_WORDS, WAKE_WORD_TIMEOUT_S

class State(Enum):
    PASSIVE = "passive"  # Listening for wake word
    ACTIVE = "active"    # Recording command after wake word

class StateMachine:
    def __init__(
        self,
        on_wake: Callable,
        on_listening: Callable,
        on_command: Callable[[str], None],
        on_error: Callable[[str], None]
    ):
        """
        Initialize state machine with callbacks.
        
        Args:
            on_wake: Called when wake word detected
            on_listening: Called when actively listening for command
            on_command: Called with transcribed command text
            on_error: Called on errors
        """
        self.state = State.PASSIVE
        self.on_wake = on_wake
        self.on_listening = on_listening
        self.on_command = on_command
        self.on_error = on_error
        
        self.active_timeout_task: Optional[asyncio.Task] = None
        
        print(f"[StateMachine] Initialized in {self.state.value} mode")
    
    def is_wake_word(self, text: str) -> bool:
        """Check if text contains wake word."""
        text_lower = text.lower().strip()
        for wake_word in WAKE_WORDS:
            if wake_word in text_lower:
                print(f"[StateMachine] Wake word '{wake_word}' detected in '{text}'")
                return True
        return False
    
    async def process_transcription(self, text: str):
        """
        Process transcribed text based on current state.
        
        In PASSIVE: Check for wake word
        In ACTIVE: Process as command
        """
        if not text:
            return
        
        if self.state == State.PASSIVE:
            # Check for wake word
            if self.is_wake_word(text):
                await self.transition_to_active()
            else:
                # Not a wake word, ignore
                print(f"[StateMachine] Ignoring passive utterance: '{text}'")
        
        elif self.state == State.ACTIVE:
            # Cancel timeout
            if self.active_timeout_task:
                self.active_timeout_task.cancel()
                self.active_timeout_task = None
            
            # Process as command
            print(f"[StateMachine] Command received: '{text}'")
            await self.on_command(text)
            
            # Return to passive
            await self.transition_to_passive()
    
    async def transition_to_active(self):
        """Transition to ACTIVE state (listening for command)."""
        if self.state == State.ACTIVE:
            return
        
        print(f"[StateMachine] Transitioning to ACTIVE")
        self.state = State.ACTIVE
        
        await self.on_wake()
        await self.on_listening()
        
        # Set timeout
        self.active_timeout_task = asyncio.create_task(self._active_timeout())
    
    async def transition_to_passive(self):
        """Transition to PASSIVE state (listening for wake word)."""
        if self.state == State.PASSIVE:
            return
        
        print(f"[StateMachine] Transitioning to PASSIVE")
        self.state = State.PASSIVE
        
        # Cancel timeout if active
        if self.active_timeout_task:
            self.active_timeout_task.cancel()
            self.active_timeout_task = None
    
    async def manual_trigger(self):
        """Manual activation (from extension's start_listening message)."""
        print(f"[StateMachine] Manual trigger")
        await self.transition_to_active()
    
    async def stop(self):
        """Stop current session and return to passive."""
        print(f"[StateMachine] Stop requested")
        await self.transition_to_passive()
    
    async def _active_timeout(self):
        """Timeout handler for ACTIVE state."""
        try:
            await asyncio.sleep(WAKE_WORD_TIMEOUT_S)
            print(f"[StateMachine] Active state timeout, returning to passive")
            await self.on_error("No command received after wake word")
            await self.transition_to_passive()
        except asyncio.CancelledError:
            pass  # Normal cancellation