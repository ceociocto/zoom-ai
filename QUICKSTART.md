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

# Install Playwright browsers
uv run playwright install chromium
```

### macOS

```bash
# Run setup script
./scripts/setup-macos.sh

# Create virtual environment and install dependencies
uv sync

# Install Playwright browsers
uv run playwright install chromium

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
