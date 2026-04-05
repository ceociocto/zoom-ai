"""
Enhanced WLK + Virtual Camera Integration with Chinese Support

Features:
- Chinese text rendering using PIL
- Smooth animations for new captions
- Multiple caption styles (Karaoke, Subtitle, Chat)
- Speaker color coding
- Modern UI design with rounded corners and shadows
"""

import asyncio
import numpy as np
import cv2
import re
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from enum import Enum

from PIL import ImageFont, ImageDraw, Image
from loguru import logger

from zoom_ai.wlk_captions import WhisperLiveKitStreamer, WLKCaptionEvent
from zoom_ai.camera import VirtualCamera


# Sentence boundary patterns for Chinese and English
SENTENCE_DELIMITERS = re.compile(r'[。！？\.!?]+')


class CaptionStyle(Enum):
    """Caption display styles."""
    SUBTITLE = "subtitle"    # Traditional subtitle at bottom
    CHAT = "chat"           # Chat-style bubbles
    KARAOKE = "karaoke"     # Karaoke-style highlighting
    MODERN = "modern"       # Modern floating card


@dataclass
class AnimationConfig:
    """Animation configuration."""
    fade_in_duration: float = 0.25  # seconds (snappier, more premium)
    slide_in: bool = True
    slide_offset: int = 40  # pixels (subtle slide effect)


@dataclass
class CaptionItem:
    """A single caption with animation state."""
    text: str
    speaker: str
    timestamp: datetime
    alpha: float = 0.0  # For fade-in animation
    y_offset: float = 0.0  # For slide-in animation

    @property
    def age(self) -> float:
        """Age of caption in seconds."""
        return (datetime.now() - self.timestamp).total_seconds()


@dataclass
class EnhancedOverlayConfig:
    """Enhanced caption overlay configuration.

    Design System: Modern/Friendly
    - Warmer dark background for comfort during long meetings
    - Distinct speaker colors for easy differentiation
    - Generous padding for readability
    - Smooth animations for premium feel
    """
    # Style
    style: CaptionStyle = CaptionStyle.MODERN

    # Layout
    position: str = "bottom"  # "top", "bottom", "center"
    max_lines: int = 5
    line_height: int = 52  # Tighter for multi-line readability
    padding: int = 24  # More breathing room
    margin_sides: int = 30

    # Font (will use Chinese font)
    font_size: int = 36
    speaker_font_size: int = 28

    # Colors (BGR) — Design System: Modern/Friendly
    background_color: tuple = (24, 24, 28)  # Softer dark, closer to modern apps
    background_alpha: float = 0.90  # More opaque for better readability
    text_color: tuple = (250, 250, 249)  # Warm white, not pure white
    speaker_colors: List[tuple] = field(default_factory=lambda: [
        (59, 130, 246),   # Blue — Speaker 1
        (139, 92, 246),   # Purple — Speaker 2
        (16, 185, 129),   # Green — Speaker 3
        (245, 158, 11),   # Orange — Speaker 4
        (236, 72, 153),   # Pink — Speaker 5
        (20, 184, 166),   # Teal — Speaker 6
    ])

    # Effects
    rounded_corners: int = 16  # Friendlier, more rounded
    shadow_blur: int = 25  # Softer shadow
    shadow_offset: Tuple[int, int] = (0, 6)  # More depth

    # Animation
    animation: Optional[AnimationConfig] = field(default_factory=AnimationConfig)


class ChineseFontLoader:
    """Loader for Chinese fonts with fallback."""

    _fonts: Dict[str, ImageFont.FreeTypeFont] = {}

    @classmethod
    def get_font(cls, size: int) -> ImageFont.FreeTypeFont:
        """Get font with specified size (cached)."""
        key = f"main_{size}"
        if key not in cls._fonts:
            cls._fonts[key] = cls._load_font(size)
        return cls._fonts[key]

    @classmethod
    def get_speaker_font(cls, size: int) -> ImageFont.FreeTypeFont:
        """Get speaker font (cached)."""
        key = f"speaker_{size}"
        if key not in cls._fonts:
            cls._fonts[key] = cls._load_font(size)
        return cls._fonts[key]

    @classmethod
    def _load_font(cls, size: int) -> ImageFont.FreeTypeFont:
        """Load font with fallback chain.

        Design System Priority:
        1. PingFang SC (macOS) — Modern, clean, excellent readability
        2. Noto Sans CJK (Linux) — Open source, widely available
        3. WQY Zenhei (Linux) — Fallback for older systems
        """
        font_paths = [
            # macOS — Design System primary: PingFang SC
            "/System/Library/Fonts/PingFang.ttc",  # Modern, clean
            "/System/Library/Fonts/Hiragino Sans GB.ttc",  # Backup
            "/System/Library/Fonts/STHeiti Light.ttc",  # Legacy fallback
            # Linux — Design System primary: Noto Sans CJK
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJKtc-Regular.otf",
            # Linux — Fallback
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            # Ultimate fallback
            None,
        ]

        for font_path in font_paths:
            try:
                if font_path:
                    font = ImageFont.truetype(font_path, size)
                    logger.debug(f"Loaded font: {font_path}")
                    return font
            except Exception:
                continue

        # Use default font
        logger.warning("Using default font - Chinese may not display correctly")
        return ImageFont.load_default()

    @classmethod
    def clear_cache(cls):
        """Clear font cache."""
        cls._fonts.clear()


class EnhancedCaptionRenderer:
    """
    Enhanced caption renderer with Chinese support and animations.
    """

    def __init__(self, config: Optional[EnhancedOverlayConfig] = None):
        """Initialize renderer."""
        self.config = config or EnhancedOverlayConfig()
        self._captions: deque = deque(maxlen=self.config.max_lines)
        self._speaker_names: Dict[str, str] = {}
        # Track current active caption for each speaker
        self._current_caption: Dict[str, CaptionItem] = {}
        # Maximum characters per caption before forced split
        self._max_caption_length = 50

    def on_caption(self, event: WLKCaptionEvent):
        """Handle new caption event - accumulates by speaker, splits at sentence boundaries."""
        speaker_id = event.speaker or "SPEAKER_0"

        # Generate friendly speaker name
        if speaker_id not in self._speaker_names:
            speaker_num = len(self._speaker_names) + 1
            self._speaker_names[speaker_id] = f"说话人 {speaker_num}"

        # Check if speaker has an active caption
        if speaker_id in self._current_caption:
            current = self._current_caption[speaker_id]
            # Append new text to current caption
            current.text += event.text

            # Check for sentence boundary or length limit
            should_split = self._should_split_caption(current.text, event.text)

            if should_split:
                # Finalize current caption
                self._captions.append(current)
                del self._current_caption[speaker_id]
                logger.info(f"[{self._speaker_names[speaker_id]}] {current.text}")
            else:
                # Caption updated in place (will be re-rendered with new text)
                pass
        else:
            # Start new caption for this speaker
            caption = CaptionItem(
                text=event.text,
                speaker=speaker_id,
                timestamp=datetime.now(),
                alpha=0.0,
                y_offset=self.config.animation.slide_offset if self.config.animation.slide_in else 0
            )
            self._current_caption[speaker_id] = caption

        # Add any finalized captions to display queue (they accumulate until speaker changes)
        for sp_id, cap in list(self._current_caption.items()):
            if sp_id != speaker_id:
                # Different speaker finished - add their caption to queue
                self._captions.append(cap)
                del self._current_caption[sp_id]
                logger.info(f"[{self._speaker_names[sp_id]}] {cap.text}")

    def _should_split_caption(self, full_text: str, new_text: str) -> bool:
        """Determine if caption should be split based on sentence boundary or length."""
        # Check length limit
        if len(full_text) >= self._max_caption_length:
            return True

        # Check if new text ends with sentence delimiter
        if SENTENCE_DELIMITERS.search(new_text):
            return True

        # Check if new text starts with a common sentence starter (after stripping)
        starters = ['但是', '然后', '所以', '因为', '如果', '不过', '而且', '另外',
                   'But', 'Then', 'So', 'Because', 'If', 'However', 'Also']
        stripped_new = new_text.strip()
        for starter in starters:
            if stripped_new.startswith(starter):
                return True

        return False

    def update_animations(self, dt: float):
        """Update animation states."""
        anim = self.config.animation
        if not anim:
            return

        # Update both finalized captions and active captions
        all_captions = list(self._captions) + list(self._current_caption.values())
        for caption in all_captions:
            # Fade in
            if caption.alpha < 1.0:
                caption.alpha = min(1.0, caption.alpha + dt / anim.fade_in_duration)

            # Slide in
            if caption.y_offset > 0:
                caption.y_offset = max(0, caption.y_offset - dt * 150)

    def _get_display_captions(self) -> List[CaptionItem]:
        """Get all captions to display (finalized + active)."""
        finalized = list(self._captions)
        active = list(self._current_caption.values())
        return finalized + active

    def render(self, frame: np.ndarray) -> np.ndarray:
        """Render captions on frame."""
        # Convert to PIL for text rendering
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img, 'RGBA')

        width, height = img.size

        # Render based on style
        if self.config.style == CaptionStyle.MODERN:
            self._render_modern(draw, width, height)
        elif self.config.style == CaptionStyle.CHAT:
            self._render_chat(draw, width, height)
        elif self.config.style == CaptionStyle.KARAOKE:
            self._render_karaoke(draw, width, height)
        else:
            self._render_subtitle(draw, width, height)

        # Convert back to OpenCV
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def _render_modern(self, draw: ImageDraw.ImageDraw, width: int, height: int):
        """Render modern floating card style."""
        captions = self._get_display_captions()
        if not captions:
            return
        line_height = self.config.line_height
        padding = self.config.padding
        margin = self.config.margin_sides

        # Calculate total height
        total_height = len(captions) * line_height + 2 * padding

        # Position
        if self.config.position == "top":
            y_base = padding + 20
        elif self.config.position == "center":
            y_base = (height - total_height) // 2
        else:
            y_base = height - total_height - padding - 20

        # Draw shadow
        shadow_y = y_base + self.config.shadow_offset[1]
        self._draw_rounded_rect(
            draw,
            (margin, shadow_y, width - margin, shadow_y + total_height),
            (0, 0, 0, 80),
            self.config.rounded_corners
        )

        # Draw background card
        bg_color = self.config.background_color
        bg_alpha = int(self.config.background_alpha * 255)
        self._draw_rounded_rect(
            draw,
            (margin, y_base, width - margin, y_base + total_height),
            (*bg_color, bg_alpha),
            self.config.rounded_corners
        )

        # Draw captions
        font = ChineseFontLoader.get_font(self.config.font_size)
        speaker_font = ChineseFontLoader.get_speaker_font(self.config.speaker_font_size)

        for i, caption in enumerate(captions):
            y = y_base + padding + i * line_height + caption.y_offset

            # Skip if outside bounds
            if y > height - 20:
                break

            # Apply fade
            alpha = int(caption.alpha * 255)
            if alpha < 255:
                # Need to create a new drawing context for alpha
                pass

            speaker_name = self._speaker_names.get(caption.speaker, caption.speaker)
            color = self._get_speaker_color(caption.speaker)

            # Draw speaker badge
            badge_text = f"{speaker_name}: "
            badge_bbox = draw.textbbox((0, 0), badge_text, font=speaker_font)
            badge_width = badge_bbox[2] - badge_bbox[0]

            # Speaker background
            self._draw_rounded_rect(
                draw,
                (margin + 10, y + 5, margin + 10 + badge_width, y + line_height - 10),
                (*color, 180),
                8
            )

            # Speaker text
            draw.text(
                (margin + 15, y + 8),
                badge_text,
                font=speaker_font,
                fill=(255, 255, 255, alpha)
            )

            # Caption text
            text_x = margin + 20 + badge_width
            draw.text(
                (text_x, y + 8),
                caption.text,
                font=font,
                fill=(*self.config.text_color, alpha)
            )

    def _render_chat(self, draw: ImageDraw.ImageDraw, width: int, height: int):
        """Render chat bubble style."""
        captions = self._get_display_captions()
        if not captions:
            return
        padding = 15
        margin = 20
        bubble_height = 50
        spacing = 15

        y_pos = height - margin - bubble_height

        for caption in reversed(captions):
            if y_pos < 50:
                break

            speaker_name = self._speaker_names.get(caption.speaker, caption.speaker)
            color = self._get_speaker_color(caption.speaker)

            # Estimate text width
            font = ChineseFontLoader.get_font(self.config.font_size - 4)
            text = f"{speaker_name}: {caption.text}"

            # Simple bubble width estimation
            bubble_width = min(width - 2 * margin, len(text) * 20 + 2 * padding)

            # Draw bubble
            self._draw_rounded_rect(
                draw,
                (margin, y_pos, margin + bubble_width, y_pos + bubble_height),
                (*color, 200),
                12
            )

            # Draw text
            draw.text(
                (margin + padding, y_pos + 12),
                text[:40] + "..." if len(text) > 40 else text,
                font=font,
                fill=(255, 255, 255)
            )

            y_pos -= bubble_height + spacing

    def _render_karaoke(self, draw: ImageDraw.ImageDraw, width: int, height: int):
        """Render karaoke style."""
        captions = self._get_display_captions()
        if not captions:
            return

        caption = captions[-1]  # Only show latest
        font = ChineseFontLoader.get_font(int(self.config.font_size * 1.5))

        text = caption.text
        speaker_name = self._speaker_names.get(caption.speaker, caption.speaker)

        # Center positioning
        y_pos = height - 150

        # Draw speaker
        speaker_font = ChineseFontLoader.get_speaker_font(self.config.speaker_font_size)
        speaker_bbox = draw.textbbox((0, 0), speaker_name, font=speaker_font)
        speaker_width = speaker_bbox[2] - speaker_bbox[0]
        draw.text(
            ((width - speaker_width) // 2, y_pos - 40),
            speaker_name,
            font=speaker_font,
            fill=self._get_speaker_color(caption.speaker)
        )

        # Draw background bar
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]

        bar_x = (width - text_width - 40) // 2
        self._draw_rounded_rect(
            draw,
            (bar_x, y_pos, bar_x + text_width + 40, y_pos + 60),
            (0, 0, 0, 200),
            10
        )

        # Draw text
        draw.text(
            ((width - text_width) // 2, y_pos + 15),
            text,
            font=font,
            fill=(255, 255, 255)
        )

    def _render_subtitle(self, draw: ImageDraw.ImageDraw, width: int, height: int):
        """Render traditional subtitle style."""
        captions = self._get_display_captions()
        if not captions:
            return
        font = ChineseFontLoader.get_font(self.config.font_size)

        # Calculate total height
        total_height = len(captions) * self.config.line_height
        y_base = height - total_height - 30

        # Draw semi-transparent background
        for i, caption in enumerate(captions):
            y = y_base + i * self.config.line_height

            # Background
            text = f"{self._speaker_names.get(caption.speaker, caption.speaker)}: {caption.text}"
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]

            bg_x = (width - text_width - 20) // 2
            draw.rectangle(
                (bg_x, y, bg_x + text_width + 20, y + self.config.line_height - 5),
                fill=(0, 0, 0, 180)
            )

            # Text
            speaker_name = self._speaker_names.get(caption.speaker, caption.speaker)
            color = self._get_speaker_color(caption.speaker)

            # Draw speaker name
            speaker_text = f"{speaker_name}: "
            draw.text(
                (bg_x + 10, y + 5),
                speaker_text,
                font=font,
                fill=color
            )

            # Draw caption text
            speaker_width = draw.textbbox((0, 0), speaker_text, font=font)[2]
            draw.text(
                (bg_x + 10 + speaker_width, y + 5),
                caption.text,
                font=font,
                fill=(255, 255, 255)
            )

    def _draw_rounded_rect(self, draw: ImageDraw.ImageDraw, coords: Tuple[int, int, int, int],
                          color: Tuple, radius: int):
        """Draw rounded rectangle."""
        x1, y1, x2, y2 = coords
        draw.rounded_rectangle(
            [(x1, y1), (x2, y2)],
            radius=radius,
            fill=color
        )

    def _get_speaker_color(self, speaker_id: str) -> Tuple:
        """Get consistent color for speaker."""
        idx = hash(speaker_id) % len(self.config.speaker_colors)
        return self.config.speaker_colors[idx]

    def clear(self):
        """Clear all captions."""
        self._captions.clear()
        self._speaker_names.clear()
        self._current_caption.clear()


class EnhancedWLKStreamer:
    """
    Enhanced WLK + Virtual Camera integration with improved rendering.
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
        overlay_config: Optional[EnhancedOverlayConfig] = None,
    ):
        """Initialize enhanced streamer."""
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

        # Enhanced overlay renderer
        self._overlay = EnhancedCaptionRenderer(overlay_config)
        self._wlk.on_caption(self._overlay.on_caption)

        # State
        self._is_running = False
        self._stream_task: Optional[asyncio.Task] = None
        self._last_time = datetime.now()

    async def start(self):
        """Start the enhanced streamer."""
        logger.info("Starting Enhanced WLK Camera Streamer...")

        # Open virtual camera
        if not self._camera.open():
            raise RuntimeError("Failed to open virtual camera")

        # Start WLK streamer
        await self._wlk.start()

        # Start frame streaming
        self._is_running = True
        self._last_time = datetime.now()
        self._stream_task = asyncio.create_task(self._stream_loop())

        logger.info("Enhanced WLK Camera Streamer started")
        logger.info(f"Camera: {self._camera.device} @ {self._camera.width}x{self._camera.height}@{self._camera.fps}fps")
        logger.info(f"Style: {self._overlay.config.style.value}")

    async def stop(self):
        """Stop the streamer."""
        logger.info("Stopping Enhanced WLK Camera Streamer...")
        self._is_running = False

        if self._stream_task:
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass

        await self._wlk.stop()
        self._camera.close()

        logger.info("Enhanced WLK Camera Streamer stopped")

    async def _stream_loop(self):
        """Stream frames with enhanced caption overlay."""
        logger.info("Enhanced frame streaming loop started")

        while self._is_running:
            try:
                # Calculate delta time for animations
                now = datetime.now()
                dt = (now - self._last_time).total_seconds()
                self._last_time = now

                # Update animations
                self._overlay.update_animations(dt)

                # Generate frame
                frame = self._generate_frame()

                if frame is not None:
                    # Apply caption overlay
                    frame_with_overlay = self._overlay.render(frame)
                    self._camera.write(frame_with_overlay)

                await asyncio.sleep(1 / self._camera.fps)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in enhanced stream loop: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(1)

        logger.info("Enhanced frame streaming loop stopped")

    def _generate_frame(self) -> Optional[np.ndarray]:
        """Generate a styled background frame."""
        frame = np.zeros((self._camera.height, self._camera.width, 3), dtype=np.uint8)

        # Create modern gradient background
        for y in range(self._camera.height):
            for x in range(self._camera.width):
                # Diagonal gradient
                r = int(30 + 50 * x / self._camera.width)
                g = int(20 + 40 * y / self._camera.height)
                b = int(50 + 60 * (x + y) / (self._camera.width + self._camera.height))
                frame[y, x] = (b, g, r)

        # Add decorative elements
        cv2.circle(
            frame,
            (self._camera.width - 100, 100),
            80,
            (60, 100, 150),
            -1
        )

        # Add title using PIL for Chinese support
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)

        font_title = ChineseFontLoader.get_font(40)
        font_sub = ChineseFontLoader.get_font(24)

        # Draw title
        draw.text((50, 40), "Zoom AI 实时字幕", font=font_title, fill=(255, 255, 255))

        # Draw status
        timestamp = datetime.now().strftime("%H:%M:%S")
        status_text = f"时间: {timestamp} | WLK: {'已连接' if self._is_running else '未连接'}"
        draw.text((50, 100), status_text, font=font_sub, fill=(200, 200, 200))

        # Style indicator
        style_text = f"字幕样式: {self._overlay.config.style.value.upper()}"
        draw.text((50, 140), style_text, font=font_sub, fill=(150, 200, 255))

        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def set_style(self, style: CaptionStyle):
        """Change caption style."""
        self._overlay.config.style = style
        logger.info(f"Caption style changed to: {style.value}")


async def test_enhanced_wlk_camera(
    wlk_server_url: str = "ws://localhost:8000/asr",
    language: str = "zh",
    diarization: bool = True,
    duration: int = 60,
    style: str = "modern",
):
    """
    Test Enhanced WLK + Virtual Camera integration.

    Args:
        wlk_server_url: WLK server WebSocket URL
        language: Language code
        diarization: Enable speaker identification
        duration: Test duration in seconds
        style: Caption style (modern, chat, karaoke, subtitle)
    """
    logger.info("Testing Enhanced WLK Camera Overlay...")

    # Create config with specified style
    config = EnhancedOverlayConfig(
        style=CaptionStyle(style),
        position="bottom",
    )

    streamer = EnhancedWLKStreamer(
        wlk_server_url=wlk_server_url,
        language=language,
        diarization=diarization,
        overlay_config=config,
    )

    try:
        await streamer.start()

        print(f"\n{'='*60}")
        print("🎥 Enhanced WLK 字幕测试已启动!")
        print(f"📢 请对着麦克风说话，查看虚拟摄像头效果")
        print(f"🎨 字幕样式: {style.upper()}")
        print(f"📍 虚拟摄像头: {streamer._camera.device}")
        print(f"⏱️  测试时长: {duration} 秒")
        print(f"{'='*60}\n")

        print("提示: 按 Ctrl+C 提前结束测试\n")

        await asyncio.sleep(duration)
        return 0

    except KeyboardInterrupt:
        print("\n✅ 测试已结束")
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
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced WLK Camera Overlay Test")
    parser.add_argument("--server-url", "-s", default="ws://localhost:8000/asr",
                        help="WLK server URL")
    parser.add_argument("--language", "-l", default="zh",
                        help="Language code")
    parser.add_argument("--duration", "-d", type=int, default=60,
                        help="Test duration in seconds")
    parser.add_argument("--style", choices=["modern", "chat", "karaoke", "subtitle"],
                        default="modern", help="Caption style")
    parser.add_argument("--no-diarization", action="store_true",
                        help="Disable speaker identification")

    args = parser.parse_args()

    asyncio.run(test_enhanced_wlk_camera(
        wlk_server_url=args.server_url,
        language=args.language,
        diarization=not args.no_diarization,
        duration=args.duration,
        style=args.style,
    ))
