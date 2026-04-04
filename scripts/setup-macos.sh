#!/bin/bash
# Setup script for Zoom AI on macOS using uv

set -e

echo "=========================================="
echo "Zoom AI Setup Script for macOS"
echo "=========================================="
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "Using uv: $(uv --version)"
echo ""

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "Homebrew not found. Installing..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Install dependencies
echo "Installing system dependencies..."
brew install \
    ffmpeg \
    portaudio

# Install OBS Virtual Camera (for virtual webcam support)
echo ""
echo "=========================================="
echo "Virtual Camera Setup (macOS)"
echo "=========================================="
echo ""
echo "macOS doesn't have built-in virtual camera support like Linux (v4l2loopback)."
echo "Recommended options:"
echo ""
echo "1. OBS Studio with Virtual Camera:"
echo "   - Install: brew install --cask obs"
echo "   - Start OBS and enable 'Start Virtual Camera'"
echo ""
echo "2. CamTwist (older but still works):"
echo "   - Install: brew install --cask camtwist"
echo ""
echo "3. Use Loopback (paid, from Rogue Amoeba)"
echo ""

# Optional: Install OBS
read -p "Install OBS Studio for virtual camera? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    brew install --cask obs
    echo "OBS installed. Start OBS and enable 'Start Virtual Camera' to use."
fi

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Create virtual environment and install dependencies:"
echo "   uv sync"
echo ""
echo "2. Copy .env.example to .env and configure:"
echo "   cp .env.example .env"
echo ""
echo "3. Install WhisperLiveKit for audio captioning (optional, recommended):"
echo "   uv pip install whisperlivekit sounddevice"
echo ""
echo "4. Start OBS and enable Virtual Camera, then test:"
echo "   uv run python -m zoom_ai.cli test-avatar"
echo ""
