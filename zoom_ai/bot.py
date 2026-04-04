"""
Zoom Bot module entry point.

This module provides the main bot functionality for running
the Zoom AI virtual avatar assistant.

Uses audio-based caption reading (WhisperLiveKit or MLX Whisper)
instead of inefficient DOM-based approaches.
"""

from zoom_ai.bot import ZoomBot, MultiInstanceBotManager, CaptionEvent
from zoom_ai.audio_captions import (
    AudioCaptionReader,
    AudioCaptionEvent,
    AudioCaptionLogger,
    AudioCapturer,
    WhisperTranscriber,
)
from zoom_ai.wlk_captions import (
    WhisperLiveKitClient,
    WhisperLiveKitStreamer,
    WLKCaptionEvent,
    WLKCaptionLogger,
    WhisperLiveKitServer,
)
from zoom_ai.wlk_camera_overlay import (
    WLKCameraStreamer,
    CaptionOverlayRenderer,
    OverlayConfig,
    ActiveSpeaker,
)

__all__ = [
    "ZoomBot",
    "MultiInstanceBotManager",
    "CaptionEvent",
    "AudioCaptionReader",
    "AudioCaptionEvent",
    "AudioCaptionLogger",
    "AudioCapturer",
    "WhisperTranscriber",
    "WhisperLiveKitClient",
    "WhisperLiveKitStreamer",
    "WLKCaptionEvent",
    "WLKCaptionLogger",
    "WhisperLiveKitServer",
    "WLKCameraStreamer",
    "CaptionOverlayRenderer",
    "OverlayConfig",
    "ActiveSpeaker",
]
