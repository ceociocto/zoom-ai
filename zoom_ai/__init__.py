"""
Zoom AI - Virtual Avatar Meeting Assistant

A server-side multi-instance virtual avatar assistant for Zoom meetings.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from zoom_ai.config import settings
from zoom_ai.camera import VirtualCamera
from zoom_ai.tts import TTSManager
from zoom_ai.avatar import AvatarRenderer

__all__ = [
    "settings",
    "VirtualCamera",
    "TTSManager",
    "AvatarRenderer",
]
