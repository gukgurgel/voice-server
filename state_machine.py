"""
State machine for wake word detection and command processing.
Two states: PASSIVE (listening for wake word) and ACTIVE (recording command).

Key fix: When the wake word and command appear in the SAME utterance
(e.g. "Hey tab, group my tabs by topic"), we extract the command portion
and process it immediately instead of discarding it.
"""

import re
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

        # Pre-compile wake word patterns for efficient matching
        # Sort by length descending so "tab whisperer" matches before "tab"
        self._wake_patterns = sorted(WAKE_WORDS, key=len, reverse=True)

        print(f"[StateMachine] Initialized in {self.state.value} mode")
        print(f"[StateMachine] Wake words: {self._wake_patterns}")

    def extract_wake_word(self, text: str) -> tuple[bool, str]:
        """
        Check if text contains a wake word and extract the remainder.

        Returns:
            (found, remainder): found=True if wake word detected,
                                remainder=text after the wake word (may be empty)

        Example:
            "hey tab group my tabs" → (True, "group my tabs")
            "hey tab"              → (True, "")
            "random speech"        → (False, "")
        """
        text_lower = text.lower().strip()

        for wake_word in self._wake_patterns:
            idx = text_lower.find(wake_word)
            if idx != -1:
                # Extract everything after the wake word
                after = text[idx + len(wake_word):].strip()
                # Clean up common filler between wake word and command
                # e.g. "hey tab, please group..." → "please group..."
                after = re.sub(r'^[,.\s!]+', '', after).strip()

                print(f"[StateMachine] Wake word '{wake_word}' found in '{text}'")
                if after:
                    print(f"[StateMachine] Remainder after wake word: '{after}'")
                return True, after

        return False, ""

    async def process_transcription(self, text: str):
        """
        Process transcribed text based on current state.

        In PASSIVE: Check for wake word. If found with trailing command,
                    process command immediately (no extra utterance needed).
        In ACTIVE:  Process as command.
        """
        if not text:
            return

        if self.state == State.PASSIVE:
            found, remainder = self.extract_wake_word(text)

            if found:
                await self.transition_to_active()

                if remainder:
                    # Wake word + command in same utterance!
                    # Process the command immediately.
                    print(f"[StateMachine] Inline command: '{remainder}'")
                    await self._handle_command(remainder)
                # else: wake word only, wait for next utterance as command

            else:
                # Not a wake word — ignore in passive mode
                print(f"[StateMachine] Ignoring passive utterance: '{text[:50]}'")

        elif self.state == State.ACTIVE:
            await self._handle_command(text)

    async def _handle_command(self, text: str):
        """Process a command and return to passive."""
        # Cancel timeout
        self._cancel_timeout()

        print(f"[StateMachine] Command: '{text}'")
        await self.on_command(text)

        # Return to passive after processing
        await self.transition_to_passive()

    async def transition_to_active(self):
        """Transition to ACTIVE state (listening for command)."""
        if self.state == State.ACTIVE:
            return

        print(f"[StateMachine] PASSIVE → ACTIVE")
        self.state = State.ACTIVE

        await self.on_wake()
        await self.on_listening()

        # Set timeout — if no command arrives, return to passive
        self.active_timeout_task = asyncio.create_task(self._active_timeout())

    async def transition_to_passive(self):
        """Transition to PASSIVE state (listening for wake word)."""
        if self.state == State.PASSIVE:
            return

        print(f"[StateMachine] ACTIVE → PASSIVE")
        self.state = State.PASSIVE
        self._cancel_timeout()

    async def manual_trigger(self):
        """Manual activation (from extension's start_listening message)."""
        print(f"[StateMachine] Manual trigger → ACTIVE")
        await self.transition_to_active()

    async def stop(self):
        """Stop current session and return to passive."""
        print(f"[StateMachine] Stop requested → PASSIVE")
        await self.transition_to_passive()

    def _cancel_timeout(self):
        """Cancel the active timeout task if running."""
        if self.active_timeout_task and not self.active_timeout_task.done():
            self.active_timeout_task.cancel()
        self.active_timeout_task = None

    async def _active_timeout(self):
        """Timeout handler for ACTIVE state."""
        try:
            await asyncio.sleep(WAKE_WORD_TIMEOUT_S)
            print(f"[StateMachine] Timeout ({WAKE_WORD_TIMEOUT_S}s), returning to passive")
            await self.on_error("No command received after wake word")
            await self.transition_to_passive()
        except asyncio.CancelledError:
            pass  # Normal cancellation when command arrives in time