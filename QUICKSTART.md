# Quick Start Guide

## 1. Prerequisites

- Python 3.10+
- Linux (for v4l2loopback) or macOS (with OBS Virtual Camera)
- FFmpeg

## 2. Installation

### Prerequisites

Install [uv](https://github.com/astral-sh/uv) - the fast Python package manager:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Linux (Ubuntu/Debian)

```bash
# Clone repository
git clone https://github.com/your-repo/zoom-ai.git
cd zoom-ai

# Run setup script (installs system deps + uv)
sudo ./scripts/setup.sh

# Create virtual environment and install dependencies
uv sync
```

### macOS

```bash
# Run setup script
./scripts/setup-macos.sh

# Create virtual environment and install dependencies
uv sync

# Start OBS and enable Virtual Camera
```

## 3. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit with your settings
nano .env
```

Key settings:
- `MEETING_ID` - Your Zoom meeting ID
- `MEETING_PASSWORD` - Meeting password (if required)
- `BOT_NAME` - Display name for the bot
- `TTS_PROVIDER` - Text-to-speech provider (edge, azure, elevenlabs)
- `AVATAR_MODEL` - Avatar type (static, sadtalker, musetalk)

## 4. Test Components

### Test Virtual Camera (Linux only)

```bash
uv run python -m zoom_ai.cli test-camera
```

This displays a test pattern on the virtual camera for 10 seconds.

### Test TTS

```bash
uv run python -m zoom_ai.cli test-tts --text "Hello, this is a test."
```

This synthesizes speech and plays it.

### Test Avatar

```bash
uv run python -m zoom_ai.cli test-avatar --duration 10
```

This streams the avatar to the virtual camera for 10 seconds.

### Test Audio Captions (Whisper)

```bash
# Install audio dependencies first
uv sync --extra audio

# Test with Whisper (captures microphone audio)
uv run python -m zoom_ai.cli test-audio-captions \
    --model base \
    --duration 60 \
    --language zh \
    --output audio_captions.txt
```

This captures system audio and transcribes it using Whisper. No Zoom meeting required.

**Whisper Model Sizes:**
- `tiny` - Fastest, least accurate (~1GB RAM)
- `base` - Good balance (~1GB RAM)
- `small` - Better accuracy (~2GB RAM)
- `medium` - High accuracy (~5GB RAM)
- `large` - Best accuracy (~10GB RAM)

### Test WhisperLiveKit (Recommended)

```bash
# Option 1: Quick install (recommended)
./scripts/install-wlk.sh      # Linux/macOS
scripts\install-wlk.bat        # Windows

# Option 2: Manual install (minimum dependencies)
uv pip install whisperlivekit sounddevice

# Step 0: Test audio capture (optional but recommended)
uv run python -m zoom_ai.test_audio 5

# Step 2a: Start WLK server (without diarization - recommended)
uv run wlk --model base --language zh

# Step 2b: Or start with diarization (requires NeMo)
uv run wlk --model base --language zh --diarization

# Step 3: Test streaming captions (terminal 2)
uv run python -m zoom_ai.cli test-wlk --duration 60

# Step 4: Test with diarization
uv run python -m zoom_ai.cli test-wlk --diarization --duration 60

# Step 5: Or auto-start server (single terminal)
uv run python -m zoom_ai.cli test-wlk --auto-server --model base --duration 60
```

**WLK Server Options:**
```bash
# View all options
uv run wlk --help

# Common options:
--model base          # Model: tiny, base, small, medium, large-v3
--language zh         # Language: zh, en, auto, etc.
--host localhost      # Server host
--port 8000           # Server port
--target-language en  # Translate to another language
--diarization         # Enable speaker identification (requires NeMo)
```

**Diarization (说话人识别) Setup:**
```bash
# Install NeMo for diarization support (~1GB download)
uv pip install "git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]"

# Start server with diarization
uv run wlk --model base --language zh --diarization

# Test with diarization
uv run python -m zoom_ai.cli test-wlk --diarization --duration 60
```

**Note:** Diarization is optional. Without it, you still get ultra-low latency transcription, just without speaker identification.

**Test Command Options:**
```bash
uv run python -m zoom_ai.cli test-wlk \
    --server-url ws://localhost:8000/asr \  # Custom server URL
    --model base \                            # Model for auto-start
    --language zh \                           # Language code
    --duration 60 \                           # Test duration
    --output captions.txt \                   # Output file
    --auto-server                             # Auto-start WLK server
```

**WhisperLiveKit Advantages:**
- Ultra-low latency streaming (< 1s)
- Automatic speaker identification (diarization)
- Real-time translation support (200 languages)
- Better accuracy than batch Whisper
- SOTA 2025 Simul-Whisper + AlignAtt policy

**macOS Notes:**
- `sounddevice` is recommended and pre-installed with the script
- If you need `pyaudio`, install portaudio first: `brew install portaudio`
- The script handles pyaudio installation failures gracefully

**Troubleshooting:**
- If connection drops immediately, check microphone permissions
- Run audio test: `uv run python -m zoom_ai.test_audio 5`
- Check WLK server logs for errors
- Try with a shorter duration first: `--duration 10`

## 5. Run the Bot

### Single Instance

```bash
uv run python -m zoom_ai.cli start \
    --meeting-id "123456789" \
    --meeting-password "secret" \
    --bot-name "AI Assistant" \
    --device /dev/video0
```

### Multiple Instances

```bash
uv run python -m zoom_ai.cli start --instances 3
```

This starts 3 bot instances using `/dev/video0`, `/dev/video1`, and `/dev/video2`.

## 6. Docker Deployment

```bash
# Build image
docker build -f docker/Dockerfile -t zoom-ai-bot .

# Run single instance
docker-compose up -d

# Scale to multiple instances
docker-compose up -d --scale bot=3
```

## 7. Verify in Zoom

1. Open Zoom and join/create a meeting
2. Select video settings
3. Choose "ZoomAI Camera" (or similar) as your camera
4. You should see the avatar!

## Troubleshooting

### Virtual camera not found

```bash
# Linux: Check v4l2loopback
lsmod | grep v4l2loopback
v4l2-ctl --list-devices

# macOS: Make sure OBS Virtual Camera is enabled in OBS
```

### Permission denied

```bash
# Add user to video group (Linux)
sudo usermod -a -G video $USER
# Log out and back in
```

### Module not found

```bash
# Reinstall dependencies with uv
uv sync

# Or recreate the environment
uv venv --python 3.10
uv sync
```

## Next Steps

- Add your own avatar image to `assets/avatar.png`
- Configure TTS API keys for better voice quality
- Set up SadTalker for lip-sync animation
- Deploy to a server for 24/7 operation

See [DEPLOYMENT.md](DEPLOYMENT.md) for production deployment guide.
