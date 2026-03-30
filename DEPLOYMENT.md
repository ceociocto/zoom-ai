# Deployment Guide

## Local Development Setup

### Linux (Ubuntu/Debian)

```bash
# Run setup script
sudo ./scripts/setup.sh

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync

# Install Playwright browsers
uv run playwright install chromium
uv run playwright install-deps chromium

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Test virtual camera
uv run python -m zoom_ai.cli test-camera
```

### macOS

```bash
# Run setup script
./scripts/setup-macos.sh

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync

# Install Playwright browsers
uv run playwright install chromium

# Configure environment
cp .env.example .env

# Start OBS and enable Virtual Camera, then test:
uv run python -m zoom_ai.cli test-avatar
```

## Docker Deployment

### Build Image

```bash
docker build -f docker/Dockerfile -t zoom-ai-bot:latest .
```

### Single Instance

```bash
# Configure environment
cp .env.example .env
# Edit .env with your meeting details

# Run container
docker-compose up -d

# View logs
docker-compose logs -f
```

### Multi-Instance

```bash
# Scale to 3 instances (using /dev/video0, /dev/video1, /dev/video2)
docker-compose up -d --scale bot=3

# View logs for specific instance
docker logs zoom-ai-bot -f
```

## Server Deployment (Production)

### Prerequisites

1. **Linux Server** (Ubuntu 22.04 recommended)
2. **GPU** (optional, for faster avatar rendering)
3. **Stable internet connection**
4. **Zoom account** (for meeting access)

### Systemd Service

Create `/etc/systemd/system/zoom-ai.service`:

```ini
[Unit]
Description=Zoom AI Virtual Avatar Bot
After=network.target

[Service]
Type=simple
User=zoomai
WorkingDirectory=/opt/zoom-ai
Environment="PATH=/opt/zoom-ai/.venv/bin"
EnvironmentFile=/opt/zoom-ai/.env
ExecStart=/opt/zoom-ai/.venv/bin/python -m zoom_ai.bot
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable zoom-ai
sudo systemctl start zoom-ai
sudo systemctl status zoom-ai
```

### Multi-Instance with Systemd

Create template service `/etc/systemd/system/zoom-ai@.service`:

```ini
[Unit]
Description=Zoom AI Bot Instance %i
After=network.target

[Service]
Type=simple
User=zoomai
WorkingDirectory=/opt/zoom-ai
Environment="PATH=/opt/zoom-ai/.venv/bin"
EnvironmentFile=/opt/zoom-ai/.env
Environment="DEVICE_INDEX=%i"
ExecStart=/opt/zoom-ai/.venv/bin/python -m zoom_ai.cli start --device /dev/video%i
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Start multiple instances:

```bash
# Start 3 instances
sudo systemctl start zoom-ai@0
sudo systemctl start zoom-ai@1
sudo systemctl start zoom-ai@2

# Enable at boot
sudo systemctl enable zoom-ai@{0..2}
```

## Cloud Deployment

### AWS EC2

```bash
# Launch GPU instance (g4dn.xlarge or similar)
# AMI: Ubuntu 22.04

# SSH into instance
ssh -i key.pem ubuntu@instance-ip

# Run setup
git clone https://github.com/your-repo/zoom-ai.git
cd zoom-ai
sudo ./scripts/setup.sh

# Install NVIDIA drivers (for GPU instance)
sudo apt install -y nvidia-driver-535

# Continue with regular setup
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
uv run playwright install chromium
```

### Google Cloud Platform

```bash
# Launch GPU VM
gcloud compute instances create zoom-ai-bot \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=100GB

# SSH and setup
gcloud compute ssh zoom-ai-bot --zone=us-central1-a
# ... continue with regular setup
```

## Troubleshooting

### Virtual Camera Issues

```bash
# Check if v4l2loopback is loaded
lsmod | grep v4l2loopback

# List camera devices
v4l2-ctl --list-devices

# Recreate devices
sudo modprobe -r v4l2loopback
sudo modprobe v4l2loopback devices=4 exclusive_caps=1
```

### Permission Issues

```bash
# Add user to video group
sudo usermod -a -G video $USER

# Log out and back in for changes to take effect
```

### Audio Issues

```bash
# List audio devices
pactl list sources

# Test audio
aplay /usr/share/sounds/alsa/Front_Center.wav
```

## Monitoring

### Health Check

```bash
# Check if service is running
systemctl is-active zoom-ai

# Check logs
journalctl -u zoom-ai -f
```

### Metrics

- Frame rate to virtual camera
- Audio synthesis latency
- Meeting connection status
- Resource usage (CPU, GPU, memory)
