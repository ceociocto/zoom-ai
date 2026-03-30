# Zoom AI 虚拟人会议助理

服务器端多实例虚拟人会议助理，可加入多个Zoom会议，以虚拟人形象参与会议。

## 特性

- **虚拟摄像头**: 使用 v4l2loopback (Linux) 创建虚拟摄像头
- **TTS语音**: 支持 Edge (免费)、Azure、ElevenLabs
- **虚拟人渲染**: 静态图片 / SadTalker 动画
- **多实例**: 单服务器运行多个 Bot 实例
- **Docker部署**: 完整的容器化支持
- **字幕读取**:
  - **DOM 方式**: 监听 Zoom web 端字幕元素
  - **Whisper 音频**: 批处理转录 (稳定)
  - **WhisperLiveKit**: 超低延迟流式转录 + 说话人识别 ⭐推荐

## 快速开始

### 1. 安装 uv

[uv](https://github.com/astral-sh/uv) 是极速的 Python 包管理器：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. 系统依赖

**Linux (Ubuntu/Debian):**
```bash
sudo ./scripts/setup.sh
```

**macOS:**
```bash
./scripts/setup-macos.sh
```

### 3. 安装依赖

```bash
# uv 会自动创建虚拟环境并安装 pyproject.toml 中的依赖
uv sync

# 安装 Playwright 浏览器
uv run playwright install chromium
```

### 4. 配置

```bash
cp .env.example .env
nano .env  # 编辑配置
```

### 5. 运行

```bash
# 测试虚拟摄像头 (Linux)
uv run python -m zoom_ai.cli test-camera

# 方式1: DOM 字幕读取 (需要加入 Zoom 会议)
uv run python -m zoom_ai.cli test-captions --meeting-id "xxx" --duration 60

# 方式2: 音频转录 (使用 Whisper，直接捕获麦克风音频)
uv run python -m zoom_ai.cli test-audio-captions --model base --duration 60

# 启动单个实例
uv run python -m zoom_ai.cli start --meeting-id "xxx" --meeting-password "xxx"

# 启动多实例
uv run python -m zoom_ai.cli start --instances 3
```

### 字幕读取功能

Zoom AI 支持三种方式读取会议字幕：

#### 方式1: DOM 监听 (需要加入 Zoom web 会议)

```python
from zoom_ai import ZoomCaptionsReader, CaptionsLogger

reader = ZoomCaptionsReader(
    meeting_id="123456789",
    display_name="AI Assistant"
)

logger = CaptionsLogger(output_file="captions.txt")
reader.on_caption(logger.on_caption)

await reader.start()
# 字幕将实时保存到文件
```

#### 方式2: Whisper 音频转录 (批处理)

```python
from zoom_ai import AudioCaptionReader, AudioCaptionLogger

reader = AudioCaptionReader(
    model_size="base",  # tiny, base, small, medium, large
    language="zh",      # zh, en, auto
)

logger = AudioCaptionLogger(output_file="audio_captions.txt")
reader.on_caption(logger.on_caption)

await reader.start()
# 直接捕获麦克风音频并转录
```

#### 方式3: WhisperLiveKit 流式转录 ⭐推荐

**超低延迟 + 说话人识别 + 实时翻译**

```bash
# 快速安装 (使用脚本)
./scripts/install-wlk.sh      # Linux/macOS
scripts\install-wlk.bat        # Windows

# 或手动安装 (最小依赖)
uv pip install whisperlivekit sounddevice

# 基础模式 (无说话人识别)
uv run wlk --model base --language zh

# 说话人识别模式 (需要额外安装 NeMo)
uv run wlk --model base --language zh --diarization

# 在另一个终端测试
uv run python -m zoom_ai.cli test-wlk --duration 60
```

**启用说话人识别:**
```bash
# 安装 NeMo (约 1GB)
uv pip install "git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]"

# 然后使用 --diarization 参数
uv run wlk --model base --language zh --diarization
```

**macOS 用户注意**: 如果 pyaudio 安装失败，可以手动安装 portaudio:
```bash
brew install portaudio
uv pip install pyaudio
```
不过 `sounddevice` 已经足够使用，pyaudio 是可选的。

```python
from zoom_ai import WhisperLiveKitStreamer, WLKCaptionLogger

streamer = WhisperLiveKitStreamer(
    server_url="ws://localhost:8000/asr",
    language="zh",
    diarization=True,  # 说话人识别
)

logger = WLKCaptionLogger(output_file="wlk_captions.txt")
streamer.on_caption(logger.on_caption)

await streamer.start()
# 超低延迟实时转录 + 自动识别说话人
```
# 直接捕获系统音频并转录
```

# 字幕将实时保存到文件
```

## 架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                         服务器 (Linux)                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │
│  │  Zoom Bot   │  │  Zoom Bot   │  │  Zoom Bot   │  ← 多实例         │
│  │  Instance 1 │  │  Instance 2 │  │  Instance N │                  │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                  │
│         │                │                │                          │
│         ▼                ▼                ▼                          │
│  ┌─────────────────────────────────────────────────────┐            │
│  │         v4l2loopback 虚拟摄像头设备                   │            │
│  │    /dev/video0, /dev/video1, /dev/video2...         │            │
│  └─────────────────────────────────────────────────────┘            │
│         ▲                                                            │
│         │                                                            │
│  ┌──────┴──────────────────────────────────────────────────┐        │
│  │  FFmpeg 推流 + 虚拟人渲染                                  │        │
│  │  (SadTalker/MuseTalk + TTS)                              │        │
│  └─────────────────────────────────────────────────────────┘        │
│                                                                     │
│  ┌─────────────────────────────────────────────────────┐            │
│  │           字幕读取 (三种方式)                          │            │
│  │  ┌───────────┐  ┌───────────┐  ┌─────────────┐      │            │
│  │  │ DOM 监听  │  │ Whisper   │  │WLK 流式     │      │            │
│  │  │          │  │ 批处理    │  │ ⭐推荐     │      │            │
│  │  └───────────┘  └───────────┘  └─────────────┘      │            │
│  └─────────────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────────────┘
```

## 项目结构

```
zoom-ai/
├── zoom_ai/
│   ├── bot/              # Zoom机器人核心
│   ├── avatar/           # 虚拟人渲染
│   ├── tts/              # 文字转语音
│   ├── camera/           # 虚拟摄像头管理
│   ├── captions.py       # DOM字幕读取
│   ├── audio_captions.py # Whisper音频转录
│   ├── wlk_captions.py   # WhisperLiveKit流式转录
│   └── config/           # 配置
├── docker/               # Docker配置
├── scripts/              # 安装脚本
├── pyproject.toml        # 项目配置 (uv)
└── .python-version       # Python版本
```

## 技术栈

| 组件 | 技术选型 |
|------|----------|
| 虚拟人动画 | SadTalker / MuseTalk |
| 虚拟摄像头 | v4l2loopback |
| 视频推流 | FFmpeg |
| TTS服务 | Azure / ElevenLabs / Edge TTS |
| 字幕读取(DOM) | Playwright |
| 字幕读取(Whisper) | OpenAI Whisper + sounddevice |
| 字幕读取(WLK) | WhisperLiveKit (Simul-Whisper + Sortformer) |
| 包管理 | uv |
| 容器化 | Docker |

## 开源参考

- [SadTalker](https://github.com/OpenTalker/SadTalker) - 音频驱动的说话人头像动画
- [v4l2loopback](https://github.com/umlaeute/v4l2loopback) - Linux虚拟摄像头
- [WhisperLiveKit](https://github.com/QuentinFuxa/WhisperLiveKit) - 超低延迟流式转录
- [Building a Zoom Bot](https://www.recall.ai/blog/how-to-build-a-zoom-bot) - Zoom机器人开发指南

## License

MIT
