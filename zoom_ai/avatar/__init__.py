"""
Virtual avatar rendering module.

Supports multiple avatar animation methods:
- SadTalker: Audio-driven single image talking face
- MuseTalk: Real-time audio-driven portrait animation
- Static: Static image/video with lip sync (simplified)
"""

import asyncio
from abc import ABC, abstractmethod
from pathlib import Path
from typing import AsyncIterator, Optional

import cv2
import numpy as np
from loguru import logger

from zoom_ai.config import settings


class AvatarRenderer(ABC):
    """Abstract base class for avatar renderers."""

    def __init__(
        self,
        source: str,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
    ):
        """
        Initialize avatar renderer.

        Args:
            source: Source image/video path.
            width: Output width.
            height: Output height.
            fps: Output FPS.
        """
        self.source = Path(source)
        self.width = width
        self.height = height
        self.fps = fps
        self._is_running = False

    @abstractmethod
    async def generate_frame(self, audio_data: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate a single frame.

        Args:
            audio_data: Optional audio data for lip sync (PCM, 16kHz).

        Returns:
            Frame as numpy array (BGR format, HxWx3).
        """
        pass

    @abstractmethod
    async def stream(self, audio_queue: Optional[asyncio.Queue] = None) -> AsyncIterator[np.ndarray]:
        """
        Stream frames continuously.

        Args:
            audio_queue: Optional queue of audio chunks for lip sync.

        Yields:
            Frames as numpy arrays.
        """
        pass

    async def start(self):
        """Start the avatar renderer."""
        self._is_running = True
        logger.info(f"Avatar renderer started: {self.__class__.__name__}")

    async def stop(self):
        """Stop the avatar renderer."""
        self._is_running = False
        logger.info(f"Avatar renderer stopped: {self.__class__.__name__}")

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to output dimensions."""
        return cv2.resize(frame, (self.width, self.height))


class StaticAvatarRenderer(AvatarRenderer):
    """
    Simple static avatar renderer.
    Displays a static image or loops a video.
    """

    def __init__(
        self,
        source: str,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
    ):
        """Initialize static avatar renderer."""
        super().__init__(source, width, height, fps)
        self._video_cap: Optional[cv2.VideoCapture] = None
        self._static_image: Optional[np.ndarray] = None
        self._frame_count = 0

    async def _load_source(self):
        """Load source image or video."""
        if not self.source.exists():
            logger.warning(f"Source not found: {self.source}, using default")
            # Create default avatar
            self._static_image = self._create_default_avatar()
            return

        if self.source.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}:
            # Load static image
            self._static_image = cv2.imread(str(self.source))
            if self._static_image is None:
                logger.error(f"Failed to load image: {self.source}")
                self._static_image = self._create_default_avatar()
        elif self.source.suffix.lower() in {'.mp4', '.avi', '.mov', '.webm'}:
            # Load video
            self._video_cap = cv2.VideoCapture(str(self.source))
            if not self._video_cap.isOpened():
                logger.error(f"Failed to open video: {self.source}")
                self._static_image = self._create_default_avatar()

    def _create_default_avatar(self) -> np.ndarray:
        """Create a default avatar image."""
        # Create gradient background
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Gradient from blue to purple
        for x in range(self.width):
            r = int(100 + (x / self.width) * 50)
            g = int(50 + (x / self.width) * 50)
            b = int(150 + (x / self.width) * 50)
            frame[:, x] = [b, g, r]

        # Add circle for "face"
        center = (self.width // 2, self.height // 2)
        radius = min(self.width, self.height) // 4
        cv2.circle(frame, center, radius, (255, 255, 255), -1)

        # Add eyes
        eye_y = center[1] - radius // 4
        eye_offset = radius // 2
        cv2.circle(frame, (center[0] - eye_offset, eye_y), radius // 5, (50, 50, 50), -1)
        cv2.circle(frame, (center[0] + eye_offset, eye_y), radius // 5, (50, 50, 50), -1)

        # Add smile
        smile_radius = radius // 2
        cv2.ellipse(
            frame,
            center,
            (smile_radius, smile_radius // 2),
            0,
            0,
            -180,
            (50, 50, 50),
            3
        )

        # Add text
        text = "AI Assistant"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, 1, 2)[0]
        text_x = (self.width - text_size[0]) // 2
        cv2.putText(frame, text, (text_x, self.height - 50), font, 1, (255, 255, 255), 2)

        return frame

    async def generate_frame(self, audio_data: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate a frame."""
        if self._static_image is not None:
            return self._resize_frame(self._static_image)

        if self._video_cap is not None:
            ret, frame = self._video_cap.read()
            if ret:
                return self._resize_frame(frame)
            else:
                # Loop video
                self._video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._video_cap.read()
                if ret:
                    return self._resize_frame(frame)

        # Fallback
        return self._resize_frame(self._create_default_avatar())

    async def stream(self, audio_queue: Optional[asyncio.Queue] = None) -> AsyncIterator[np.ndarray]:
        """Stream frames."""
        await self._load_source()

        while self._is_running:
            frame = await self.generate_frame()
            yield frame
            await asyncio.sleep(1 / self.fps)

    async def start(self):
        """Start the renderer."""
        await self._load_source()
        await super().start()

    async def stop(self):
        """Stop the renderer."""
        await super().stop()
        if self._video_cap is not None:
            self._video_cap.release()
            self._video_cap = None


class SadTalkerRenderer(AvatarRenderer):
    """
    SadTalker-based avatar renderer.
    Audio-driven single image talking face animation.

    Requires: SadTalker installation and models.
    """

    def __init__(
        self,
        source: str,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        model_path: Optional[str] = None,
    ):
        """Initialize SadTalker renderer."""
        super().__init__(source, width, height, fps)
        self.model_path = model_path
        self._model = None
        self._source_image: Optional[np.ndarray] = None

    async def _load_model(self):
        """Load SadTalker model."""
        try:
            import torch
            # Lazy import SadTalker
            # This requires SadTalker to be installed
            logger.info("Loading SadTalker model...")
            # Model loading would go here
            # self._model = load_sadtalker_model(self.model_path)
            logger.warning("SadTalker integration requires manual setup")
        except ImportError:
            logger.error("SadTalker not installed. Use 'pip install -e .[sadtalker]' and install SadTalker separately.")
            raise

    async def _load_source_image(self):
        """Load source image for animation."""
        if not self.source.exists():
            logger.warning(f"Source not found: {self.source}")
            # Create default
            self._source_image = self._create_default_avatar()
            return

        self._source_image = cv2.imread(str(self.source))
        if self._source_image is None:
            logger.error(f"Failed to load source image: {self.source}")
            self._source_image = self._create_default_avatar()

    def _create_default_avatar(self) -> np.ndarray:
        """Create default avatar (same as StaticAvatarRenderer)."""
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for x in range(self.width):
            r = int(100 + (x / self.width) * 50)
            g = int(50 + (x / self.width) * 50)
            b = int(150 + (x / self.width) * 50)
            frame[:, x] = [b, g, r]
        center = (self.width // 2, self.height // 2)
        radius = min(self.width, self.height) // 4
        cv2.circle(frame, center, radius, (255, 255, 255), -1)
        eye_y = center[1] - radius // 4
        eye_offset = radius // 2
        cv2.circle(frame, (center[0] - eye_offset, eye_y), radius // 5, (50, 50, 50), -1)
        cv2.circle(frame, (center[0] + eye_offset, eye_y), radius // 5, (50, 50, 50), -1)
        return frame

    async def generate_frame(self, audio_data: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate animated frame based on audio."""
        if self._source_image is None:
            return self._create_default_avatar()

        # TODO: Implement actual SadTalker inference
        # For now, return source image
        return self._resize_frame(self._source_image)

    async def stream(self, audio_queue: Optional[asyncio.Queue] = None) -> AsyncIterator[np.ndarray]:
        """Stream animated frames."""
        await self._load_source_image()

        # Buffer audio chunks
        audio_buffer = []

        while self._is_running:
            # Get audio from queue if available
            if audio_queue and not audio_queue.empty():
                while not audio_queue.empty():
                    audio_buffer.append(await audio_queue.get())

            # Generate frame with audio sync
            audio_data = np.concatenate(audio_buffer) if audio_buffer else None
            frame = await self.generate_frame(audio_data)

            # Clear buffer after processing
            audio_buffer.clear()

            yield frame
            await asyncio.sleep(1 / self.fps)

    async def start(self):
        """Start the renderer."""
        await self._load_source_image()
        # await self._load_model()  # Uncomment when model is set up
        await super().start()


class AvatarRendererFactory:
    """Factory for creating avatar renderers."""

    @staticmethod
    def create(
        renderer_type: str = "static",
        source: Optional[str] = None,
        **kwargs
    ) -> AvatarRenderer:
        """
        Create an avatar renderer instance.

        Args:
            renderer_type: Type of renderer ("static", "sadtalker", "musetalk").
            source: Source image/video path.
            **kwargs: Additional arguments for the renderer.

        Returns:
            AvatarRenderer instance.
        """
        source = source or settings.avatar_image_path

        if renderer_type == "static":
            return StaticAvatarRenderer(source, **kwargs)
        elif renderer_type == "sadtalker":
            return SadTalkerRenderer(source, **kwargs)
        # elif renderer_type == "musetalk":
        #     return MuseTalkRenderer(source, **kwargs)
        else:
            logger.warning(f"Unknown renderer type: {renderer_type}, using static")
            return StaticAvatarRenderer(source, **kwargs)
