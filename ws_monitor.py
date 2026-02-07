#!/usr/bin/env python3
"""
WebSocket monitor for the Tab Whisperer voice server.
Connects and logs every JSON message with timestamps.

Usage:
    python3 ws_monitor.py                    # just watch messages
    python3 ws_monitor.py --log session.jsonl # also save to file
    python3 ws_monitor.py --interactive      # watch + send test messages

The server supports one client at a time, so if the extension is
connected, this will replace it. Run this INSTEAD of the extension
for debugging, or modify the server to support multiple clients.
"""

import asyncio
import websockets
import json
import sys
import os
from datetime import datetime
from argparse import ArgumentParser

# ANSI colors for terminal readability
COLORS = {
    "status":    "\033[36m",    # cyan
    "wake":      "\033[33;1m",  # bold yellow
    "listening": "\033[33m",    # yellow
    "command":   "\033[32;1m",  # bold green
    "error":     "\033[31;1m",  # bold red
    "sent":      "\033[35m",    # magenta (our outgoing messages)
    "reset":     "\033[0m",
}


def colorize(msg_type: str, text: str) -> str:
    color = COLORS.get(msg_type, "")
    reset = COLORS["reset"]
    return f"{color}{text}{reset}"


def format_message(direction: str, data: dict) -> str:
    """Format a message for display."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    msg_type = data.get("type", "unknown")

    if direction == "recv":
        arrow = "◄──"
        color_key = msg_type
    else:
        arrow = "──►"
        color_key = "sent"

    # Compact display for common types
    if msg_type == "command":
        detail = f'  text: "{data.get("text", "")}"'
    elif msg_type == "error":
        detail = f'  message: "{data.get("message", "")}"'
    elif msg_type == "status":
        detail = f'  model={data.get("model")} wake_word="{data.get("wake_word")}"'
    else:
        # Show full JSON for anything else
        extra = {k: v for k, v in data.items() if k != "type"}
        detail = f"  {json.dumps(extra)}" if extra else ""

    line = f"[{timestamp}] {arrow} {msg_type}{detail}"
    return colorize(color_key, line)


class WSMonitor:
    def __init__(self, url: str, log_file: str = None, interactive: bool = False):
        self.url = url
        self.log_file = log_file
        self.interactive = interactive
        self.log_handle = None
        self.message_count = 0

    async def run(self):
        if self.log_file:
            self.log_handle = open(self.log_file, "a")
            print(f"Logging to: {self.log_file}")

        print(f"Connecting to {self.url} ...")
        print(f"{'─' * 60}")

        try:
            async with websockets.connect(self.url) as ws:
                print(colorize("status", f"Connected to {self.url}"))
                print(f"{'─' * 60}")

                if self.interactive:
                    # Run receiver and sender concurrently
                    await asyncio.gather(
                        self.receive_loop(ws),
                        self.interactive_sender(ws)
                    )
                else:
                    await self.receive_loop(ws)

        except ConnectionRefusedError:
            print(colorize("error", f"Connection refused. Is the server running?"))
            print(f"  Start it with: python3 server.py")
        except KeyboardInterrupt:
            pass
        finally:
            if self.log_handle:
                self.log_handle.close()
            print(f"\n{'─' * 60}")
            print(f"Session ended. {self.message_count} messages received.")

    async def receive_loop(self, ws):
        """Receive and display all messages from the server."""
        try:
            async for raw in ws:
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    print(f"[??] Non-JSON: {raw[:100]}")
                    continue

                self.message_count += 1
                print(format_message("recv", data))

                # Log to file as JSONL (one JSON object per line)
                if self.log_handle:
                    log_entry = {
                        "ts": datetime.now().isoformat(),
                        "direction": "recv",
                        "data": data
                    }
                    self.log_handle.write(json.dumps(log_entry) + "\n")
                    self.log_handle.flush()

        except websockets.exceptions.ConnectionClosed as e:
            print(colorize("error", f"Connection closed: {e}"))

    async def interactive_sender(self, ws):
        """
        Interactive mode: read commands from stdin and send them.
        Runs in a thread to not block the event loop.
        """
        loop = asyncio.get_running_loop()

        HELP = """
Commands (type and press Enter):
  wake         → send {"type": "start_listening"}  (manual trigger)
  stop         → send {"type": "stop_listening"}
  ack <text>   → send {"type": "ack", "result": "<text>"}
  raw <json>   → send raw JSON string
  help         → show this help
  quit         → exit
"""
        print(colorize("sent", HELP))

        while True:
            try:
                # Read from stdin in thread pool (non-blocking)
                line = await loop.run_in_executor(
                    None, lambda: input(colorize("sent", "send> "))
                )
            except EOFError:
                break

            line = line.strip()
            if not line:
                continue

            msg = None

            if line == "wake":
                msg = {"type": "start_listening"}
            elif line == "stop":
                msg = {"type": "stop_listening"}
            elif line.startswith("ack "):
                msg = {"type": "ack", "result": line[4:]}
            elif line.startswith("raw "):
                try:
                    msg = json.loads(line[4:])
                except json.JSONDecodeError as e:
                    print(colorize("error", f"Invalid JSON: {e}"))
                    continue
            elif line == "help":
                print(colorize("sent", HELP))
                continue
            elif line in ("quit", "exit", "q"):
                break
            else:
                print(colorize("error", f"Unknown command: '{line}'. Type 'help'."))
                continue

            if msg:
                raw = json.dumps(msg)
                await ws.send(raw)
                print(format_message("send", msg))

                if self.log_handle:
                    log_entry = {
                        "ts": datetime.now().isoformat(),
                        "direction": "send",
                        "data": msg
                    }
                    self.log_handle.write(json.dumps(log_entry) + "\n")
                    self.log_handle.flush()


def main():
    parser = ArgumentParser(description="Monitor Tab Whisperer WebSocket messages")
    parser.add_argument(
        "--url", default="ws://localhost:8765",
        help="WebSocket URL (default: ws://localhost:8765)"
    )
    parser.add_argument(
        "--log", metavar="FILE",
        help="Save messages to JSONL file (append mode)"
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true",
        help="Interactive mode: also send test messages"
    )
    args = parser.parse_args()

    monitor = WSMonitor(args.url, log_file=args.log, interactive=args.interactive)
    asyncio.run(monitor.run())


if __name__ == "__main__":
    main()