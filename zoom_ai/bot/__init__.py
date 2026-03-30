"""
Zoom Bot - Main bot implementation for joining and participating in Zoom meetings.

Supports multiple instances with separate virtual cameras and audio devices.
"""

import asyncio
import signal
from pathlib import Path
from typing import Optional, Callable, AsyncIterator

from loguru import logger

from zoom_ai.avatar import AvatarRendererFactory
from zoom_ai.camera import VirtualCamera, FrameBuffer
from zoom_ai.config import settings, setup_logging
from zoom_ai.tts import TTSManager
from zoom_ai.captions import ZoomCaptionsReader, CaptionEvent, CaptionsLogger


class ZoomBot:
    """
    Zoom Bot instance for joining meetings with virtual avatar.
    """

    def __init__(
        self,
        meeting_id: Optional[str] = None,
        meeting_password: Optional[str] = None,
        bot_name: Optional[str] = None,
        device_index: Optional[int] = None,
    ):
        """
        Initialize Zoom Bot.

        Args:
            meeting_id: Zoom meeting ID.
            meeting_password: Zoom meeting password.
            bot_name: Display name for the bot.
            device_index: Virtual camera device index.
        """
        self.meeting_id = meeting_id or settings.meeting_id
        self.meeting_password = meeting_password or settings.meeting_password
        self.bot_name = bot_name or settings.bot_name
        self.device_index = device_index if device_index is not None else settings.device_index

        # Components
        self._camera: Optional[VirtualCamera] = None
        self._avatar = AvatarRendererFactory.create(
            renderer_type=settings.avatar_model,
            source=settings.avatar_image_path,
            width=settings.output_width,
            height=settings.output_height,
            fps=settings.output_fps,
        )
        self._tts = TTSManager()
        self._captions_reader = ZoomCaptionsReader(
            meeting_id=self.meeting_id,
            meeting_password=self.meeting_password,
            display_name=self.bot_name,
            headless=settings.browser_headless,
        )
        self._captions_logger = CaptionsLogger(output_file=f"./logs/captions_{self.bot_name.replace(' ', '_')}.txt")

        # Runtime state
        self._is_running = False
        self._frame_buffer = FrameBuffer()
        self._captions: list[CaptionEvent] = []

        logger.info(f"Zoom Bot initialized: {self.bot_name} @ /dev/video{self.device_index}")

    async def start(self):
        """Start the bot and join the meeting."""
        logger.info(f"Starting Zoom Bot: {self.bot_name}")

        # Setup signal handlers
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.stop()))

        # Initialize virtual camera
        self._camera = VirtualCamera(
            device=f"/dev/video{self.device_index}",
            width=settings.output_width,
            height=settings.output_height,
            fps=settings.output_fps,
        )

        if not self._camera.open():
            logger.error("Failed to open virtual camera")
            return False

        # Start avatar renderer
        await self._avatar.start()

        # Start frame streaming task
        self._is_running = True
        self._stream_task = asyncio.create_task(self._stream_frames())

        # Start captions reader
        self._captions_reader.on_caption(self._on_caption_received)
        await self._captions_reader.start()

        # Join Zoom meeting
        await self._join_meeting()

        logger.info(f"Zoom Bot started: {self.bot_name}")
        return True

    async def stop(self):
        """Stop the bot and leave the meeting."""
        if not self._is_running:
            return

        logger.info(f"Stopping Zoom Bot: {self.bot_name}")
        self._is_running = False

        # Cancel tasks
        if hasattr(self, '_stream_task'):
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass

        # Leave meeting
        await self._leave_meeting()

        # Stop captions reader
        await self._captions_reader.stop()

        # Stop components
        await self._avatar.stop()
        if self._camera:
            self._camera.close()

        logger.info(f"Zoom Bot stopped: {self.bot_name}")

    async def _join_meeting(self):
        """
        Join the Zoom meeting.

        This is a placeholder for the actual Zoom integration.
        Options for implementation:
        1. Zoom Meeting SDK (requires license)
        2. Playwright/Selenium automation of Zoom web client
        3. Third-party service (Recall.ai, etc.)
        """
        logger.info("Joining Zoom meeting...")
        logger.warning("Zoom meeting integration not yet implemented")
        logger.info(f"Meeting ID: {self.meeting_id}")
        logger.info(f"Bot name: {self.bot_name}")
        logger.info(f"Virtual camera: /dev/video{self.device_index}")

        # TODO: Implement actual Zoom join logic
        # Example using Playwright:
        # async with async_playwright() as p:
        #     browser = await p.chromium.launch()
        #     context = await browser.new_context(
        #         permissions=["camera", "microphone"],
        #     )
        #     page = await context.new_page()
        #     await page.goto(f"https://zoom.us/j/{self.meeting_id}")
        #     # ... interact with page to join meeting

    async def _leave_meeting(self):
        """Leave the Zoom meeting."""
        logger.info("Leaving Zoom meeting...")
        # TODO: Implement actual Zoom leave logic

    async def _stream_frames(self):
        """
        Stream avatar frames to virtual camera.
        """
        logger.debug("Frame streaming started")

        try:
            async for frame in self._avatar.stream():
                if not self._is_running:
                    break

                # Write frame to virtual camera
                if self._camera:
                    self._camera.write(frame)

                # Small delay to maintain frame rate
                await asyncio.sleep(1 / settings.output_fps)

        except asyncio.CancelledError:
            logger.debug("Frame streaming cancelled")
        except Exception as e:
            logger.error(f"Error in frame streaming: {e}")
        finally:
            logger.debug("Frame streaming stopped")

    async def speak(self, text: str):
        """
        Speak text through the bot.

        Args:
            text: Text to speak.
        """
        logger.info(f"Speaking: {text[:100]}...")

        # Generate audio
        audio_path = await self._tts.speak(text)

        # TODO: Stream audio to Zoom meeting
        # This requires audio routing/virtual microphone

    async def listen(self, duration: float = 5.0) -> str:
        """
        Listen to meeting audio and transcribe.

        Args:
            duration: Listen duration in seconds.

        Returns:
            Transcribed text.
        """
        # TODO: Implement audio capture and transcription
        # Requires virtual microphone setup
        logger.warning("Listen not yet implemented")
        return ""

    def _on_caption_received(self, event: CaptionEvent):
        """Handle received caption event."""
        self._captions.append(event)
        self._captions_logger.on_caption(event)
        logger.debug(f"Caption received: [{event.speaker or 'Unknown'}] {event.text[:50]}...")

    async def get_captions(self) -> AsyncIterator[CaptionEvent]:
        """
        Get an async iterator of caption events.

        Usage:
            async for caption in bot.get_captions():
                print(f"{caption.speaker}: {caption.text}")
        """
        while self._is_running:
            if self._captions:
                yield self._captions.pop(0)
            else:
                await asyncio.sleep(0.1)

    def get_recent_captions(self, count: int = 10) -> list[CaptionEvent]:
        """
        Get the most recent captions.

        Args:
            count: Number of recent captions to return.

        Returns:
            List of recent caption events.
        """
        return self._captions[-count:] if self._captions else []

    def get_all_captions(self) -> list[CaptionEvent]:
        """Get all collected captions."""
        return self._captions.copy()

    def clear_captions(self):
        """Clear collected captions."""
        self._captions.clear()

    async def read_captions_for_duration(self, duration: float) -> list[CaptionEvent]:
        """
        Read captions for a specific duration.

        Args:
            duration: Duration in seconds.

        Returns:
            List of caption events collected during the duration.
        """
        return await self._captions_reader.read_captions_for_duration(duration)

    async def run_forever(self):
        """Run the bot until stopped."""
        await self.start()

        try:
            while self._is_running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()


class MultiInstanceBotManager:
    """
    Manages multiple Zoom Bot instances.
    """

    def __init__(self, num_instances: int = 1):
        """
        Initialize multi-instance manager.

        Args:
            num_instances: Number of bot instances to run.
        """
        self.num_instances = num_instances
        self._bots: list[ZoomBot] = []

    async def start_all(self):
        """Start all bot instances."""
        logger.info(f"Starting {self.num_instances} bot instances...")

        for i in range(self.num_instances):
            bot = ZoomBot(
                meeting_id=settings.meeting_id,
                meeting_password=settings.meeting_password,
                bot_name=f"{settings.bot_name} #{i+1}",
                device_index=i,
            )
            self._bots.append(bot)
            asyncio.create_task(bot.start())

        logger.info(f"All {self.num_instances} bot instances started")

    async def stop_all(self):
        """Stop all bot instances."""
        logger.info("Stopping all bot instances...")

        for bot in self._bots:
            await bot.stop()

        self._bots.clear()
        logger.info("All bot instances stopped")

    async def run_forever(self):
        """Run all bots until stopped."""
        await self.start_all()

        try:
            while any(bot._is_running for bot in self._bots):
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop_all()
