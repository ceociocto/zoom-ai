#!/bin/bash
# Setup script for Zoom AI on Linux (Ubuntu/Debian) using uv

set -e

echo "=========================================="
echo "Zoom AI Setup Script"
echo "=========================================="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "This script requires sudo privileges for system packages."
    echo "Please run with: sudo ./scripts/setup.sh"
    exit 1
fi

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
else
    echo "Cannot detect OS. Exiting."
    exit 1
fi

echo "Detected OS: $OS"
echo ""

# Install system dependencies
echo "Installing system dependencies..."
apt-get update

# Python and development tools
apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-dev \
    build-essential \
    curl

# Virtual camera (v4l2loopback)
apt-get install -y \
    v4l2loopback-dkms \
    v4l2loopback-utils \
    linux-modules-extra-$(uname -r)

# Video/audio processing
apt-get install -y \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libportaudio2 \
    libportaudiocpp0 \
    alsa-utils

# Browser dependencies (for Playwright)
apt-get install -y \
    wget \
    gnupg \
    ca-certificates \
    fonts-liberation \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libatspi2.0-0 \
    libcups2 \
    libdbus-1-3 \
    libdrm2 \
    libgbm1 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libwayland-client0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxkbcommon0 \
    libxrandr2 \
    xdg-utils

# Other utilities
apt-get install -y \
    git \
    vim

echo ""
echo "System dependencies installed successfully."

# Load v4l2loopback module
echo ""
echo "Loading v4l2loopback module..."
modprobe v4l2loopback devices=4 exclusive_caps=1

# Make module load on boot
if ! grep -q "v4l2loopback" /etc/modules; then
    echo "v4l2loopback" >> /etc/modules
fi

# Create modules configuration
cat > /etc/modprobe.d/v4l2loopback.conf << 'EOF'
options v4l2loopback devices=4 exclusive_caps=1 card_label="ZoomAI Camera"
EOF

echo ""
echo "v4l2loopback configured."
echo ""

# List devices
echo "Available camera devices:"
v4l2-ctl --list-devices

# Install uv as the regular user (not as root)
echo ""
echo "=========================================="
echo "Installing uv package manager..."
echo "=========================================="

# Get the username of the sudo user
REAL_USER=${SUDO_USER:-$USER}
REAL_HOME=$(eval echo ~$REAL_USER)

echo ""
echo "Installing uv for user: $REAL_USER"

# Check if uv is already installed
if ! sudo -u $REAL_USER bash -c "command -v uv &> /dev/null"; then
    sudo -u $REAL_USER bash -c "curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "uv installed successfully!"
else
    echo "uv is already installed: $(sudo -u $REAL_USER bash -c "uv --version")"
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
echo "2. Install Playwright browsers:"
echo "   uv run playwright install chromium"
echo ""
echo "3. Copy .env.example to .env and configure:"
echo "   cp .env.example .env"
echo ""
echo "4. Add user to video group (logout and back in after this):"
echo "   sudo usermod -a -G video \$USER"
echo ""
echo "5. Test the virtual camera:"
echo "   uv run python -m zoom_ai.cli test-camera"
echo ""
