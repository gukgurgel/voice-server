## Tab Whisperer — Voice Server

Local Python server that listens to the microphone, detects a wake word, transcribes voice commands with Whisper, and sends them to the browser extension over WebSocket.

```
Mic → VAD (Silero) → Whisper (whisper.cpp) → WebSocket → Extension
```

### Quick Setup

```bash
./setup.sh      # clones whisper.cpp, builds with CMake, downloads model, installs deps
python3 server.py
```

### Setup Requirements

#### Internet Required (First Run)
1. **Silero VAD**: Auto-downloads (~2.5 MB) on first `python3 server.py` run
2. **Whisper Model**: Downloaded by `setup.sh` via whisper.cpp's model script
3. **whisper.cpp**: Cloned from GitHub and built with CMake

#### Build Dependencies
- Python 3.8+
- CMake (build whisper.cpp)
- A C/C++ compiler (gcc/clang)
- A working microphone

#### Offline Operation
After initial setup, the server works completely offline:
- VAD cached at: `~/.cache/torch/hub/`
- Whisper model at: `./models/ggml-base.en.bin`
- whisper.cpp binary at: `./whisper.cpp/build/bin/whisper-cli`

### Troubleshooting
- **"Downloading..." on first run** — normal, Silero VAD auto-downloads once
- **No internet?** Run `setup.sh` on a connected machine, copy entire directory
- **Firewall issues?** Pre-download VAD: `python3 -c "import torch; torch.hub.load('snakers4/silero-vad', 'silero_vad')"`
- **cmake not found?** Install it: `brew install cmake` (macOS) or `sudo apt install cmake` (Ubuntu)

---

### WebSocket Monitor (`ws_monitor.py`)

A debugging tool that connects to the voice server and displays every WebSocket message in real time with color-coded output and timestamps. Useful for verifying the server is sending the right JSON to the extension, or for testing the voice pipeline without loading the extension at all.

#### Usage

```bash
# Watch all messages flowing through the WebSocket
python3 ws_monitor.py

# Watch + save the entire session to a file for later analysis
python3 ws_monitor.py --log session.jsonl

# Interactive mode: watch + send test messages to the server
python3 ws_monitor.py --interactive
```

#### Example output

```
[14:32:01.443] ◄── status  model=base.en wake_word="hey fox"
[14:32:05.891] ◄── wake
[14:32:05.893] ◄── listening
[14:32:08.217] ◄── command  text: "group my tabs by topic"
```

#### Interactive mode commands

| Command | Sends | Purpose |
|---------|-------|---------|
| `wake` | `{"type": "start_listening"}` | Manual trigger (skip wake word) |
| `stop` | `{"type": "stop_listening"}` | Cancel current listening session |
| `ack <text>` | `{"type": "ack", "result": "<text>"}` | Simulate extension acknowledgment |
| `raw <json>` | *(the raw JSON you type)* | Send any arbitrary message |

#### Log file format

With `--log`, messages are saved as JSONL (one JSON object per line):

```json
{"ts": "2025-02-08T14:32:05.891", "direction": "recv", "data": {"type": "wake"}}
{"ts": "2025-02-08T14:32:08.217", "direction": "recv", "data": {"type": "command", "text": "group my tabs by topic"}}
```

You can process these with standard tools:

```bash
# Pretty-print the whole session
cat session.jsonl | jq .

# Show only command messages
cat session.jsonl | jq 'select(.data.type == "command")'

# Count messages by type
cat session.jsonl | jq -r '.data.type' | sort | uniq -c
```

#### Note on single-client mode

The voice server accepts one WebSocket client at a time. If the extension is already connected and you start the monitor, it will replace the extension's connection. Use the monitor **instead of** the extension during debugging, not alongside it.