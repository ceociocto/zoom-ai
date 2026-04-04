# Zoom AI Virtual Assistant - Dockerfile
FROM ubuntu:22.04

# 设置非交互模式
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    # Python
    python3.10 \
    python3.10-venv \
    python3-pip \
    python3-dev \
    # 虚拟摄像头
    v4l2loopback-dkms \
    v4l2loopback-utils \
    linux-modules-extra-$(uname -r) \
    # 视频/音频
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libportaudio2 \
    libportaudiocpp0 \
    ffmpeg \
    # 其他工具
    curl \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

# 创建虚拟摄像头设备
RUN modprobe v4l2loopback devices=4 exclusive_caps=1 || true

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .
COPY pyproject.toml .

# 安装Python依赖
RUN pip3 install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY zoom_ai/ ./zoom_ai/
COPY assets/ ./assets/
COPY config/ ./config/

# 创建必要的目录
RUN mkdir -p /app/logs /app/tmp

# 暴露端口 (如需Web界面)
EXPOSE 8080

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import zoom_ai.health" || exit 1

# 默认命令
CMD ["python3", "-m", "zoom_ai.bot"]
