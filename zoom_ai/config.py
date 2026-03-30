"""
Configuration management for Zoom AI.
Loads settings from environment variables and .env file.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env file
load_dotenv()

class Settings(BaseSettings):
    """Application settings."""

    # Application
    app_name: str = "Zoom AI Assistant"
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_path: str = Field(default="./logs", alias="LOG_PATH")

    # Zoom API
    zoom_api_key: Optional[str] = Field(default=None, alias="ZOOM_API_KEY")
    zoom_api_secret: Optional[str] = Field(default=None, alias="ZOOM_API_SECRET")
    zoom_sdk_jwt: Optional[str] = Field(default=None, alias="ZOOM_SDK_JWT")

    # Meeting credentials
    meeting_id: Optional[str] = Field(default=None, alias="MEETING_ID")
    meeting_password: Optional[str] = Field(default=None, alias="MEETING_PASSWORD")
    bot_name: str = Field(default="AI Assistant", alias="BOT_NAME")

    # TTS Configuration
    tts_provider: str = Field(default="edge", alias="TTS_PROVIDER")
    azure_tts_key: Optional[str] = Field(default=None, alias="AZURE_TTS_KEY")
    azure_tts_region: str = Field(default="eastasia", alias="AZURE_TTS_REGION")
    elevenlabs_api_key: Optional[str] = Field(default=None, alias="ELEVENLABS_API_KEY")
    tts_voice: str = Field(default="en-US-AriaNeural", alias="TTS_VOICE")

    # Avatar Configuration
    avatar_model: str = Field(default="sadtalker", alias="AVATAR_MODEL")
    avatar_image_path: str = Field(default="./assets/avatar.png", alias="AVATAR_IMAGE_PATH")
    avatar_video_path: str = Field(default="./assets/avatar_idle.mp4", alias="AVATAR_VIDEO_PATH")

    # Virtual Camera
    virtual_camera_device: str = Field(default="/dev/video0", alias="VIRTUAL_CAMERA_DEVICE")
    v4l2loopback_devices: int = Field(default=4, alias="V4L2LOOPBACK_DEVICES")

    # Video Output
    output_width: int = Field(default=1280, alias="OUTPUT_WIDTH")
    output_height: int = Field(default=720, alias="OUTPUT_HEIGHT")
    output_fps: int = Field(default=30, alias="OUTPUT_FPS")

    # Device index for multi-instance
    device_index: int = Field(default=0, alias="DEVICE_INDEX")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            return "INFO"
        return v.upper()

    @field_validator("avatar_model")
    @classmethod
    def validate_avatar_model(cls, v: str) -> str:
        """Validate avatar model choice."""
        valid_models = {"sadtalker", "musetalk", "static"}
        if v.lower() not in valid_models:
            return "static"
        return v.lower()

    @property
    def virtual_camera_device_auto(self) -> str:
        """Get virtual camera device based on device index."""
        return f"/dev/video{self.device_index}"

    @property
    def log_dir(self) -> Path:
        """Get log directory as Path object."""
        return Path(self.log_path)

    model_config = SettingsConfigDict(
        extra="ignore",
        case_sensitive=False
    )


# Global settings instance
settings = Settings()


def setup_logging():
    """Setup logging configuration."""
    import sys
    from loguru import logger

    settings.log_dir.mkdir(parents=True, exist_ok=True)

    # Remove default handler
    logger.remove()

    # Console handler
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )

    # File handler
    logger.add(
        settings.log_dir / "zoom_ai_{time:YYYY-MM-DD}.log",
        level=settings.log_level,
        rotation="00:00",
        retention="30 days",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    )

    return logger
