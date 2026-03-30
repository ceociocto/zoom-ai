#!/bin/bash
# WhisperLiveKit Quick Install Script
# Usage: ./scripts/install-wlk.sh

set -e

echo "=========================================="
echo "  WhisperLiveKit Quick Install"
echo "=========================================="
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "✅ uv is installed"

# Install WhisperLiveKit
echo ""
echo "📦 Installing WhisperLiveKit..."
uv pip install whisperlivekit

# Install audio capture dependencies
echo ""
echo "📦 Installing audio capture dependencies..."
echo "Installing sounddevice (cross-platform)..."
uv pip install sounddevice

echo ""
echo "Installing pyaudio (optional, requires portaudio)..."
set +e  # Don't exit on error
uv pip install pyaudio 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ pyaudio installed successfully"
else
    echo "⚠️  pyaudio installation failed (this is OK, sounddevice will be used)"
    echo "   To install pyaudio on macOS, run: brew install portaudio"
fi
set -e

# Ask about diarization support
echo ""
echo "=========================================="
echo "  Diarization Support (Speaker ID)"
echo "=========================================="
echo ""
echo "Do you want to install diarization support?"
echo "This requires NVIDIA NeMo (large download ~1GB)"
echo ""
read -p "Install diarization? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "📦 Installing NeMo for diarization..."
    uv pip install "git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]"
    echo "✅ Diarization support installed"
else
    echo ""
    echo "⚠️  Skipping diarization. You can still use WLK without --diarization flag"
fi

# Verify installation
echo ""
echo "🔍 Verifying installation..."
if uv run wlk --help &> /dev/null; then
    echo ""
    echo "=========================================="
    echo "  Installation Complete!"
    echo "=========================================="
    echo ""
    echo "Quick start:"
    echo "  # Start server (without diarization)"
    echo "  uv run wlk --model base --language zh"
    echo ""
    echo "  # Start server (with diarization, if installed)"
    echo "  uv run wlk --model base --language zh --diarization"
    echo ""
    echo "  # Test with zoom-ai"
    echo "  uv run python -m zoom_ai.cli test-wlk --duration 60"
else
    echo "❌ Installation verification failed"
    exit 1
fi

echo ""
