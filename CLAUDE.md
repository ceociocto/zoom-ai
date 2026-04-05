# Claude AI Context — Zoom AI

## Project Overview

Zoom AI 虚拟人会议助理 — A server-side multi-instance virtual human meeting assistant that can join multiple Zoom meetings with a virtual avatar.

Key features:
- Virtual camera using v4l2loopback (Linux)
- TTS support (Edge, Azure, ElevenLabs, GLM)
- Virtual human rendering (static images / SadTalker animation)
- Real-time caption reading via MLX Whisper or WhisperLiveKit
- Docker deployment support

## Project Structure

```
zoom_ai/
├── bot/              # Zoom robot core
├── avatar/           # Virtual human rendering
├── tts/              # Text-to-speech services
├── camera/           # Virtual camera management
├── captions.py       # DOM subtitle reading
├── audio_captions.py # Whisper audio transcription
├── wlk_captions.py   # WhisperLiveKit streaming transcription
├── wlk_enhanced_overlay.py  # Enhanced caption overlay renderer
└── wlk_tts_overlay.py        # WLK + TTS integration
```

## Design System

**Always read DESIGN.md before making visual or UI changes to the caption overlay.**

The caption overlay in `wlk_enhanced_overlay.py` implements a complete design system with:
- Modern/Friendly aesthetic optimized for video meetings
- Specific color palette (BGR values for PIL/OpenCV)
- Typography stack prioritizing PingFang SC for Chinese
- Spacing, layout, and animation standards
- 4 caption styles: Modern, Chat, Karaoke, Subtitle

All font choices, colors, spacing, rounded corners, shadows, and animation timing are defined in DESIGN.md.
Do not deviate without explicit user approval.

When working on the overlay renderer, reference `EnhancedOverlayConfig` for the current design system values.

## Tech Stack

- **Language:** Python 3.10+
- **Package Manager:** uv
- **Virtual Camera:** v4l2loopback (Linux)
- **Video Processing:** FFmpeg, OpenCV, PIL
- **Transcription:**
  - MLX Whisper (local, Apple Silicon optimized)
  - WhisperLiveKit (streaming, ultra-low latency)
- **TTS:** Edge TTS, Azure, ElevenLabs, GLM API

## Development Quick Commands

```bash
# Install dependencies
uv sync

# Test virtual camera (Linux)
uv run python -m zoom_ai.cli test-camera

# Test audio transcription
uv run python -m zoom_ai.cli test-audio-captions --model base --duration 60

# Test WLK streaming (requires WLK server running)
uv run python -m zoom_ai.cli test-wlk --duration 60

# Test enhanced caption overlay with different styles
uv run python -m zoom_ai.cli test-wlk-enhanced --style modern --duration 60
uv run python -m zoom_ai.cli test-wlk-enhanced --style chat --duration 60
uv run python -m zoom_ai.cli test-wlk-enhanced --style karaoke --duration 60
uv run python -m zoom_ai.cli test-wlk-enhanced --style subtitle --duration 60

# Start single bot instance
uv run python -m zoom_ai.cli start --meeting-id "xxx" --meeting-password "xxx"

# Start multiple instances
uv run python -m zoom_ai.cli start --instances 3
```

## Configuration

Configuration is managed via `.env` file. Copy from `.env.example`:

```bash
cp .env.example .env
nano .env
```

Key environment variables:
- `GLM_TTS_API_KEY` — GLM TTS API key for text-to-speech
- `WLK_SERVER_URL` — WhisperLiveKit server URL (default: ws://localhost:8000/asr)

## Caption Overlay Testing

To test the caption overlay with real-time streaming:

1. Start WLK server in terminal 1:
   ```bash
   uv run wlk --model base --language zh
   ```

2. Run enhanced overlay test in terminal 2:
   ```bash
   uv run python -m zoom_ai.cli test-wlk-enhanced --style modern --duration 60
   ```

3. Speak into microphone — captions will appear in virtual camera output

## Testing

The project includes several test scripts for development:

- `test_chinese_camera.py` — Test Chinese character rendering
- `test_microphone.py` — Test microphone input
- `test_wlk_*.py` — Various WLK integration tests
- `whisper_camera_overlay.py` — Whisper + camera overlay test
