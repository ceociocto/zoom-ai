# WLK + Virtual Camera 集成指南

## 功能概述

将 WhisperLiveKit (WLK) 的实时语音识别与说话人识别功能集成到虚拟摄像头输出中，实现：

1. **实时语音识别**：超低延迟转录会议音频
2. **说话人识别**：自动识别不同发言者
3. **字幕叠加**：将识别结果叠加到虚拟摄像头输出画面上

## 架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        WLK + Virtual Camera 集成                         │
│                                                                         │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────────────────┐  │
│  │   System    │───▶│   WLK Server │───▶│   Caption Events         │  │
│  │   Audio     │    │ (WebSocket)  │    │  [Speaker, Text, Time]   │  │
│  └─────────────┘    └──────────────┘    └──────────┬───────────────┘  │
│                                                 │                       │
│                                                 ▼                       │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────────────────┐  │
│  │   Avatar    │───▶│   Overlay    │───▶│   Virtual Camera         │  │
│  │  Renderer   │    │   Renderer   │    │   /dev/video0            │  │
│  └─────────────┘    └──────────────┘    └──────────┬───────────────┘  │
│                                                 │                       │
│                                                 ▼                       │
│                                          ┌─────────────┐               │
│                                          │   Zoom      │               │
│                                          │   Meeting   │               │
│                                          └─────────────┘               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 使用方法

### 1. 启动 WLK 服务器（带说话人识别）

```bash
# 安装 NeMo（说话人识别需要）
uv pip install "git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]"

# 启动 WLK 服务器
uv run wlk --model base --language zh --diarization
```

### 2. 测试集成功能

```bash
# 在另一个终端运行测试
uv run python -m zoom_ai.cli test-wlk-camera --diarization --duration 60
```

### 3. 在 Zoom 中使用

1. 打开 Zoom 会议设置
2. 选择 "虚拟摄像头" (Virtual Camera) 作为摄像头
3. 开启麦克风（让系统捕获会议音频）
4. 开始说话，你将看到：
   - 虚拟摄像头画面带有实时字幕叠加
   - 不同说话者用不同颜色标识
   - `[Speaker 1] 大家好...`
   - `[Speaker 2] 你好...`

## Python API 使用

### 基础使用

```python
import asyncio
from zoom_ai.wlk_camera_overlay import WLKCameraStreamer

async def main():
    # 创建集成流
    streamer = WLKCameraStreamer(
        wlk_server_url="ws://localhost:8000/asr",
        language="zh",
        diarization=True,  # 启用说话人识别
    )

    await streamer.start()

    # 运行 60 秒
    await asyncio.sleep(60)

    await streamer.stop()

asyncio.run(main())
```

### 自定义字幕样式

```python
from zoom_ai.wlk_camera_overlay import (
    WLKCameraStreamer,
    OverlayConfig,
)

# 创建自定义配置
config = OverlayConfig(
    font_scale=1.0,           # 字体大小
    thickness=2,              # 字体粗细
    background_color=(0, 0, 0),  # 背景色 (BGR)
    text_color=(255, 255, 255),  # 文字颜色
    position="top",           # 位置: "top" 或 "bottom"
    max_lines=8,              # 最大显示行数
)

streamer = WLKCameraStreamer(
    wlk_server_url="ws://localhost:8000/asr",
    language="zh",
    diarization=True,
    overlay_config=config,
)
```

### 集成到 Bot 中

```python
from zoom_ai.bot import ZoomBot
from zoom_ai.wlk_camera_overlay import WLKCameraStreamer

class EnhancedZoomBot(ZoomBot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 创建 WLK + Camera 流
        self._wlk_streamer = WLKCameraStreamer(
            wlk_server_url="ws://localhost:8000/asr",
            language="zh",
            diarization=True,
            camera_device=f"/dev/video{self.device_index}",
        )

        # 设置头像帧回调
        avatar = self._avatar
        self._wlk_streamer.set_avatar_frame_callback(
            lambda: avatar.get_frame()
        )

    async def start(self):
        # 启动 WLK 流
        await self._wlk_streamer.start()
        # 启动原始 bot
        await super().start()
```

## 关键组件说明

### WLKCameraStreamer

主集成类，协调音频捕获、WLK 连接和虚拟摄像头输出。

- `wlk_server_url`: WLK 服务器 WebSocket URL
- `language`: 识别语言 (zh, en, auto 等)
- `diarization`: 是否启用说话人识别
- `camera_device`: 虚拟摄像头设备路径

### CaptionOverlayRenderer

字幕叠加渲染器，负责在视频帧上绘制字幕。

- `on_caption(event)`: 处理新的字幕事件
- `render(frame)`: 将字幕叠加到视频帧

### OverlayConfig

字幕叠加配置。

- 字体设置：大小、粗细、颜色
- 布局设置：位置、行数、间距
- 说话人颜色：自动为每个说话人分配不同颜色

## 注意事项

1. **说话人识别需要 NeMo**：安装 NeMo 约需 1GB 空间
2. **音频源配置**：确保系统麦克风能够捕获 Zoom 会议音频
   - Linux: 使用 `pavucontrol` 配置音频路由
   - macOS: 使用 Loopback 或 BlackHole
3. **虚拟摄像头**：需要预装 v4l2loopback (Linux) 或 pyvirtualcam (macOS)

## 故障排查

### 没有字幕输出
- 检查 WLK 服务器是否运行：`ps aux | grep wlk`
- 检查 WebSocket 连接：查看日志中的连接错误
- 检查音频捕获：确保麦克风有权限

### 说话人识别不工作
- 确认安装了 NeMo: `pip list | grep nemo`
- 确认启动时使用了 `--diarization` 参数
- NeMo 首次运行会下载模型，需要网络连接

### 虚拟摄像头不显示
- Linux: `sudo modprobe v4l2loopback devices=4 exclusive_caps=1`
- macOS: 确保启用了 OBS Virtual Camera
