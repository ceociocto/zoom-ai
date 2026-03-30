"""
WhisperLiveKit + Virtual Camera Integration

Streams audio to WLK for transcription with speaker identification,
and overlays the results on virtual camera output.
"""

import asyncio
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque

import numpy as np
import cv2
from loguru import logger

from zoom_ai.wlk_captions import WhisperLiveKitStreamer, WLKCaptionEvent
from zoom_ai.camera import VirtualCamera
from zoom_ai.audio_captions import AudioCapturer


@dataclass
class ActiveSpeaker:
    """Track active speaker with their recent captions."""
    speaker_id: str
    display_name: str
    last_caption: str = ""
    last_caption_time: Optional[datetime] = None
    caption_count: int = 0

    @property
    def is_active(self) -> bool:
        """Check if speaker is still active (within last 10 seconds)."""
        if self.last_caption_time is None:
            return False
        return (datetime.now() - self.last_caption_time) < timedelta(seconds=10)


@dataclass
class OverlayConfig:
    """Configuration for caption overlay on video."""
    # Font settings
    font_face: int = cv2.FONT_HERSHEY_SIMPLEX
    font_scale: float = 0.8
    thickness: int = 2

    # Colors (BGR)
    background_color: tuple = (0, 0, 0)  # Black
    text_color: tuple = (255, 255, 255)  # White
    speaker_colors: List[tuple] = field(default_factory=lambda: [
        (255, 107, 107),  # Red
        (78, 205, 196),   # Teal
        (255, 230, 109),  # Yellow
        (26, 83, 92),     # Dark Cyan
        (247, 255, 247),  # White-ish
        (255, 121, 198),  # Pink
        (133, 193, 233),  # Blue
    ])

    # Layout
    padding: int = 10
    line_height: int = 40
    max_lines: int = 6
    position: str = "bottom"  # "top" or "bottom"


class CaptionOverlayRenderer:
    """
    Renders caption overlays on video frames.
    """

    def __init__(self, config: Optional[OverlayConfig] = None):
        """
        Initialize overlay renderer.

        Args:
            config: Overlay configuration.
        """
        self.config = config or OverlayConfig()
        self._speakers: Dict[str, ActiveSpeaker] = {}
        self._recent_captions: deque = deque(maxlen=self.config.max_lines)

    def on_caption(self, event: WLKCaptionEvent):
        """
        Handle a new caption event.

        Args:
            event: Caption event from WLK.
        """
        speaker_id = event.speaker or "Unknown"

        # Update or create speaker
        if speaker_id not in self._speakers:
            self._speakers[speaker_id] = ActiveSpeaker(
                speaker_id=speaker_id,
                display_name=speaker_id
            )

        speaker = self._speakers[speaker_id]
        speaker.last_caption = event.text
        speaker.last_caption_time = datetime.now()
        speaker.caption_count += 1

        # Add to recent captions (for display)
        self._recent_captions.append({
            "speaker": speaker_id,
            "text": event.text,
            "time": datetime.now(),
        })

        logger.info(f"[Overlay] [{speaker_id}] {event.text}")

    def render(self, frame: np.ndarray) -> np.ndarray:
        """
        Render caption overlay on frame.

        Args:
            frame: Input video frame (BGR).

        Returns:
            Frame with overlay.
        """
        frame = frame.copy()
        height, width = frame.shape[:2]

        # Get active captions from recent history
        active_captions = list(self._recent_captions)
        if not active_captions:
            return frame

        # Calculate overlay dimensions
        text_lines = []
        for cap in active_captions:
            text_lines.append(f"[{cap['speaker']}] {cap['text']}")

        if not text_lines:
            return frame

        # Draw semi-transparent background
        overlay_height = len(text_lines) * self.config.line_height + 2 * self.config.padding
        y_start = height - overlay_height if self.config.position == "bottom" else 0

        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (0, y_start),
            (width, y_start + overlay_height),
            self.config.background_color,
            -1
        )
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Draw text lines
        y = y_start + self.config.padding + 30
        for i, line in enumerate(text_lines):
            speaker = active_captions[i]["speaker"]
            color = self._get_speaker_color(speaker)

            # Draw speaker name in color
            speaker_end = line.index("]") + 1
            speaker_text = line[:speaker_end]

            # Get text size for background
            (text_w, text_h), _ = cv2.getTextSize(
                speaker_text,
                self.config.font_face,
                self.config.font_scale,
                self.config.thickness
            )

            # Draw text
            cv2.putText(
                frame,
                speaker_text,
                (self.config.padding, y),
                self.config.font_face,
                self.config.font_scale,
                color,
                self.config.thickness
            )

            # Draw caption text in white
            caption_text = line[speaker_end:]
            cv2.putText(
                frame,
                caption_text,
                (self.config.padding + text_w + 10, y),
                self.config.font_face,
                self.config.font_scale,
                self.config.text_color,
                self.config.thickness
            )

            y += self.config.line_height

        return frame

    def _get_speaker_color(self, speaker_id: str) -> tuple:
        """Get consistent color for speaker."""
        idx = hash(speaker_id) % len(self.config.speaker_colors)
        return self.config.speaker_colors[idx]

    def clear(self):
        """Clear all speakers and captions."""
        self._speakers.clear()
        self._recent_captions.clear()


class WLKCameraStreamer:
    """
    Integrates WLK transcription with virtual camera output.

    Captures audio, sends to WLK for transcription, and overlays
    speaker-identified captions on the virtual camera stream.
    """

    def __init__(
        self,
        wlk_server_url: str = "ws://localhost:8000/asr",
        language: str = "zh",
        diarization: bool = True,
        camera_device: Optional[str] = None,
        camera_width: int = 1280,
        camera_height: int = 720,
        camera_fps: int = 30,
        overlay_config: Optional[OverlayConfig] = None,
    ):
        """
        Initialize WLK Camera Streamer.

        Args:
            wlk_server_url: WLK server WebSocket URL.
            language: Language code for transcription.
            diarization: Enable speaker identification.
            camera_device: Virtual camera device path.
            camera_width: Camera frame width.
            camera_height: Camera frame height.
            camera_fps: Camera frame rate.
            overlay_config: Caption overlay configuration.
        """
        self.wlk_server_url = wlk_server_url
        self.language = language
        self.diarization = diarization

        # Virtual camera
        self._camera = VirtualCamera(
            device=camera_device,
            width=camera_width,
            height=camera_height,
            fps=camera_fps,
        )

        # WLK streamer
        self._wlk = WhisperLiveKitStreamer(
            server_url=wlk_server_url,
            language=language,
            diarization=diarization,
        )

        # Overlay renderer
        self._overlay = CaptionOverlayRenderer(overlay_config)
        self._wlk.on_caption(self._overlay.on_caption)

        # State
        self._is_running = False
        self._stream_task: Optional[asyncio.Task] = None
        self._test_frame_generator: Optional[asyncio.Task] = None

    async def start(self):
        """Start the integrated streamer."""
        logger.info("Starting WLK Camera Streamer...")

        # Open virtual camera
        if not self._camera.open():
            raise RuntimeError("Failed to open virtual camera")

        # Start WLK streamer
        await self._wlk.start()

        # Start frame streaming
        self._is_running = True
        self._stream_task = asyncio.create_task(self._stream_loop())
        self._test_frame_generator = asyncio.create_task(self._generate_test_frames())

        logger.info("WLK Camera Streamer started")
        logger.info(f"Camera: {self._camera.device} @ {self._camera.width}x{self._camera.height}@{self._camera.fps}fps")
        logger.info(f"WLK: {self.wlk_server_url} (diarization={self.diarization})")

    async def stop(self):
        """Stop the streamer."""
        logger.info("Stopping WLK Camera Streamer...")
        self._is_running = False

        if self._stream_task:
            self._stream_task.cancel()
        if self._test_frame_generator:
            self._test_frame_generator.cancel()

        try:
            await asyncio.gather(self._stream_task, self._test_frame_generator, return_exceptions=True)
        except asyncio.CancelledError:
            pass

        await self._wlk.stop()
        self._camera.close()

        logger.info("WLK Camera Streamer stopped")

    async def _stream_loop(self):
        """Stream frames with caption overlay to virtual camera."""
        logger.info("Frame streaming loop started")

        while self._is_running:
            try:
                # Get frame from avatar/test generator
                # In production, this would come from AvatarRenderer
                frame = self._get_test_frame()

                if frame is not None:
                    # Apply caption overlay
                    frame_with_overlay = self._overlay.render(frame)

                    # Write to virtual camera
                    self._camera.write(frame_with_overlay)

                await asyncio.sleep(1 / self._camera.fps)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in stream loop: {e}")
                await asyncio.sleep(1)

        logger.info("Frame streaming loop stopped")

    def _get_test_frame(self) -> Optional[np.ndarray]:
        """
        Get a test frame for streaming.

        In production, replace this with actual avatar frames.
        """
        # Create a simple gradient background
        frame = np.zeros((self._camera.height, self._camera.width, 3), dtype=np.uint8)

        # Create gradient
        for y in range(self._camera.height):
            color = int(255 * y / self._camera.height)
            frame[y, :] = (50, color // 2, 100)

        # Add some text
        cv2.putText(
            frame,
            "Zoom AI - Virtual Camera",
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

        return frame

    async def _generate_test_frames(self):
        """Background task for generating test frames (if needed)."""
        while self._is_running:
            await asyncio.sleep(0)

    def set_avatar_frame_callback(self, callback):
        """
        Set a callback to get avatar frames.

        Args:
            callback: Async function that returns np.ndarray frame.
        """
        self._get_avatar_frame = callback


async def test_wlk_camera_overlay(
    wlk_server_url: str = "ws://localhost:8000/asr",
    language: str = "zh",
    diarization: bool = True,
    duration: int = 60,
):
    """
    Test WLK + Virtual Camera integration.

    This demonstrates streaming audio to WLK, getting transcriptions
    with speaker identification, and overlaying them on the virtual
    camera output.
    """
    logger.info("Testing WLK Camera Overlay...")
    logger.info(f"WLK Server: {wlk_server_url}")
    logger.info(f"Language: {language}")
    logger.info(f"Diarization: {diarization}")
    logger.info(f"Duration: {duration}s")

    streamer = WLKCameraStreamer(
        wlk_server_url=wlk_server_url,
        language=language,
        diarization=diarization,
    )

    try:
        await streamer.start()
        logger.info(f"\n{'='*50}")
        logger.info("Streaming started!")
        logger.info("Speak into your microphone to see captions")
        logger.info(f"Check your virtual camera: {streamer._camera.device}")
        logger.info(f"{'='*50}\n")

        await asyncio.sleep(duration)

        return 0

    except KeyboardInterrupt:
        print("\nTest interrupted")
        return 0
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        await streamer.stop()


if __name__ == "__main__":
    import sys

    url = sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:8000/asr"
    duration = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    diarization = "--diarization" in sys.argv

    asyncio.run(test_wlk_camera_overlay(
        wlk_server_url=url,
        duration=duration,
        diarization=diarization,
    ))
