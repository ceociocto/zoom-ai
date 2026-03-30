"""
Zoom Bot module entry point.

This module provides the main bot functionality for running
the Zoom AI virtual avatar assistant.
"""

from zoom_ai.bot import ZoomBot, MultiInstanceBotManager
from zoom_ai.captions import ZoomCaptionsReader, CaptionEvent, CaptionsLogger
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
    "ZoomCaptionsReader",
    "CaptionEvent",
    "CaptionsLogger",
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
