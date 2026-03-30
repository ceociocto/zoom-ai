# Zoom AI 虚拟人会议助理

服务器端多实例虚拟人会议助理，可加入多个Zoom会议，以虚拟人形象参与会议。

## 特性

- **虚拟摄像头**: 使用 v4l2loopback (Linux) 创建虚拟摄像头
- **TTS语音**: 支持 Edge (免费)、Azure、ElevenLabs
- **虚拟人渲染**: 静态图片 / SadTalker 动画
- **多实例**: 单服务器运行多个 Bot 实例
- **Docker部署**: 完整的容器化支持

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

# 启动单个实例
uv run python -m zoom_ai.cli start --meeting-id "xxx" --meeting-password "xxx"

# 启动多实例
uv run python -m zoom_ai.cli start --instances 3
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
| 包管理 | uv |
| 容器化 | Docker |

## 开源参考

- [SadTalker](https://github.com/OpenTalker/SadTalker) - 音频驱动的说话人头像动画
- [v4l2loopback](https://github.com/umlaeute/v4l2loopback) - Linux虚拟摄像头
- [Building a Zoom Bot](https://www.recall.ai/blog/how-to-build-a-zoom-bot) - Zoom机器人开发指南

## License

MIT
