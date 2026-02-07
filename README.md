## Setup Requirements

### Internet Required (First Run)
1. **Silero VAD**: Auto-downloads (~2.5 MB) on first `python3 server.py` run
2. **Whisper Model**: Manual download required (see setup.sh)

### Offline Operation
After initial setup, the server works completely offline:
- VAD cached at: `~/.cache/torch/hub/`
- Whisper model at: `./models/ggml-base.en.bin`

### Quick Setup
```bash
./setup.sh  # Handles everything
```

### Troubleshooting
- **"Downloading..."** on first run is normal (VAD auto-download)
- **No internet?** Run setup.sh on connected machine, copy entire directory
- **Firewall issues?** Pre-download VAD: `python3 -c "import torch; torch.hub.load('snakers4/silero-vad', 'silero_vad')"`