"""
Whisper + Virtual Camera Integration

Uses Whisper batch transcription with speaker identification overlay on virtual camera.
"""
import asyncio
import numpy as np
import cv2
from typing import Optional
from datetime import datetime, timedelta
from collections import deque

from PIL import ImageFont, ImageDraw, Image

from zoom_ai.audio_captions import AudioCaptionReader, AudioCaptionEvent
from zoom_ai.camera import VirtualCamera


class WhisperCameraOverlay:
    """
    Integrates Whisper transcription with virtual camera output.
    """

    # Class-level font cache
    _chinese_font = None

    @classmethod
    def _get_chinese_font(cls):
        """Get or load Chinese font."""
        if cls._chinese_font is None:
            font_paths = [
                "/System/Library/Fonts/STHeiti Light.ttc",
                "/System/Library/Fonts/STHeiti Medium.ttc",
                "/System/Library/Fonts/Hiragino Sans GB.ttc",
            ]
            for font_path in font_paths:
                try:
                    cls._chinese_font = ImageFont.truetype(font_path, 40)
                    print(f"✅ Loaded Chinese font: {font_path}")
                    break
                except Exception as e:
                    print(f"⚠️  Failed to load {font_path}: {e}")

            if cls._chinese_font is None:
                print("⚠️  Using default font - Chinese may not display")
                cls._chinese_font = ImageFont.load_default()

        return cls._chinese_font
    """
    Integrates Whisper transcription with virtual camera output.
    """

    def __init__(
        self,
        model_size: str = "base",
        language: str = "zh",
        camera_device: Optional[str] = None,
        camera_width: int = 1280,
        camera_height: int = 720,
        camera_fps: int = 30,
    ):
        """Initialize the integrated streamer."""
        self.model_size = model_size
        self.language = language

        # Virtual camera
        self._camera = VirtualCamera(
            device=camera_device,
            width=camera_width,
            height=camera_height,
            fps=camera_fps,
        )

        # Whisper transcription
        self._transcriber = AudioCaptionReader(
            model_size=model_size,
            language=language,
            chunk_duration_ms=5000,  # 5 second chunks
        )

        # Caption state
        self._captions: deque = deque(maxlen=5)
        self._is_running = False

    def on_caption(self, event: AudioCaptionEvent):
        """Handle new caption event."""
        self._captions.append({
            "text": event.text,
            "time": event.timestamp,
        })
        print(f"📺 [Overlay CALLBACK] {event.text[:30]}...")  # Debug

    async def start(self):
        """Start the integrated streamer."""
        # Start transcription
        print("🔧 Setting up caption callback...")
        self._transcriber.on_caption(self.on_caption)
        print("✅ Callback registered, starting transcriber...")
        await self._transcriber.start()
        print("✅ Transcriber started!")

        # Open virtual camera
        if not self._camera.open():
            raise RuntimeError("Failed to open virtual camera")

        self._is_running = True

    async def stop(self):
        """Stop the streamer."""
        self._is_running = False
        await self._transcriber.stop()
        self._camera.close()

    async def run(self, duration: int = 60):
        """Run for specified duration."""
        await self.start()

        print(f"\n{'='*60}")
        print(f"Streaming to virtual camera: {self._camera.device}")
        print(f"Speak to see captions on camera!")
        print(f"Running for {duration} seconds...")
        print(f"{'='*60}\n")

        try:
            for i in range(duration * self._camera.fps):
                frame = self._generate_frame()
                self._camera.write(frame)
                await asyncio.sleep(1 / self._camera.fps)
        finally:
            await self.stop()

    def _generate_frame(self) -> np.ndarray:
        """Generate a frame with caption overlay."""
        # Create gradient background
        frame = np.zeros((self._camera.height, self._camera.width, 3), dtype=np.uint8)

        for y in range(self._camera.height):
            color = int(255 * y / self._camera.height)
            frame[y, :] = (50, color // 3, 100)

        # Add title with OpenCV (English only)
        cv2.putText(
            frame,
            "Zoom AI - Whisper Captions",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(
            frame,
            f"Time: {timestamp}",
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            2
        )

        # Draw caption overlay using PIL for Chinese support
        if self._captions:
            overlay_height = len(self._captions) * 50 + 40
            y_start = self._camera.height - overlay_height

            # Semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (0, y_start),
                (self._camera.width, self._camera.height),
                (0, 0, 0),
                -1
            )
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            # Use PIL for Chinese text
            frame = self._draw_captions_with_pil(frame, y_start)

        return frame

    def _draw_captions_with_pil(self, frame: np.ndarray, y_start: int) -> np.ndarray:
        """Draw captions using PIL for Chinese text support."""
        # Convert to PIL image
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)

        # Get cached font
        font = self._get_chinese_font()

        # Draw captions
        y = y_start + 40
        for cap in list(self._captions):
            text = cap["text"][:40]  # Limit text length
            draw.text((20, y - 30), text, font=font, fill=(0, 255, 255))  # Yellow
            y += 50

        # Convert back to OpenCV
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


async def test_whisper_camera(duration: int = 30):
    """Test Whisper + Virtual Camera integration."""
    streamer = WhisperCameraOverlay(
        model_size="base",
        language="zh",
    )

    try:
        await streamer.run(duration=duration)
        return 0
    except KeyboardInterrupt:
        print("\nTest interrupted")
        return 0
    finally:
        await streamer.stop()


if __name__ == "__main__":
    import sys
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    asyncio.run(test_whisper_camera(duration=duration))
