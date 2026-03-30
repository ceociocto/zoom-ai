"""
Zoom Meeting Captions Reader

Uses Playwright to capture live captions from Zoom web client.
"""

import asyncio
from typing import AsyncIterator, Optional, Callable
from dataclasses import dataclass
from datetime import datetime

from playwright.async_api import async_playwright, Browser, Page, BrowserContext
from loguru import logger


@dataclass
class CaptionEvent:
    """A caption event from the meeting."""
    text: str
    speaker: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class ZoomCaptionsReader:
    """
    Reads live captions from Zoom meeting via web client automation.

    Captions are extracted from the Zoom web client's DOM elements.
    """

    # CSS selectors for Zoom web client caption elements
    CAPTION_SELECTORS = [
        # Live captions - more specific selectors
        "[data-testid*='live-caption']",
        "[data-testid*='cc-text']",
        "[data-testid*='transcript-text']",
        # Web client specific
        ".captions-container .caption-line",
        ".live-transcription-text",
        ".closed-caption-text",
        "#captions .caption-text",
        "[aria-live='polite'][aria-label*='caption']",
        "[aria-live='polite'][aria-label*='transcript']",
        # General fallbacks
        "[class*='caption-text']",
        "[class*='transcript-text']",
    ]

    # Text to filter out (false positives)
    FILTER_WORDS = {
        "preview", "loading", "字幕", "subtitles", "cc",
        "closed caption", "live transcription", "enable",
        "disable", "show", "hide", "off", "on",
    }

    def __init__(
        self,
        meeting_id: str,
        meeting_password: Optional[str] = None,
        display_name: str = "AI Assistant",
        headless: bool = False,
        debug: bool = False,
    ):
        """
        Initialize the captions reader.

        Args:
            meeting_id: Zoom meeting ID.
            meeting_password: Optional meeting password.
            display_name: Display name for the participant.
            headless: Run browser in headless mode.
            debug: Enable debug logging.
        """
        self.meeting_id = meeting_id
        self.meeting_password = meeting_password
        self.display_name = display_name
        self.headless = headless
        self.debug = debug

        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._is_running = False
        self._caption_task: Optional[asyncio.Task] = None

        # Callback for caption events
        self._on_caption: Optional[Callable[[CaptionEvent], None]] = None

    def on_caption(self, callback: Callable[[CaptionEvent], None]):
        """Register a callback for caption events."""
        self._on_caption = callback

    async def start(self):
        """Start the captions reader and join the meeting."""
        logger.info(f"Starting captions reader for meeting {self.meeting_id}")

        playwright = await async_playwright().start()

        # Launch browser
        self._browser = await playwright.chromium.launch(
            headless=self.headless,
            args=[
                "--use-fake-ui-for-media-stream",
                "--enable-features=WebRTC-H264WithOpenH264FFmpeg",
            ],
        )

        # Create context with media permissions
        self._context = await self._browser.new_context(
            permissions=["camera", "microphone"],
            viewport={"width": 1280, "height": 720},
        )

        self._page = await self._context.new_page()

        # Navigate to meeting
        await self._join_meeting()

        # Start caption monitoring
        self._is_running = True
        self._caption_task = asyncio.create_task(self._monitor_captions())

        logger.info("Captions reader started")

    async def stop(self):
        """Stop the captions reader."""
        logger.info("Stopping captions reader")

        self._is_running = False

        if self._caption_task:
            self._caption_task.cancel()
            try:
                await self._caption_task
            except asyncio.CancelledError:
                pass

        if self._page:
            await self._page.close()
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()

        logger.info("Captions reader stopped")

    async def _join_meeting(self):
        """Join the Zoom meeting."""
        logger.info(f"Joining Zoom meeting: {self.meeting_id}")

        # Navigate to Zoom meeting URL
        meeting_url = f"https://zoom.us/wc/join/{self.meeting_id}"
        await self._page.goto(meeting_url, wait_until="domcontentloaded", timeout=30000)

        # Wait for page to load
        await asyncio.sleep(3)

        # Try multiple ways to join
        joined = False

        # Method 1: Try "Join from Browser" button
        if not joined:
            try:
                # Various text patterns for the button
                join_patterns = [
                    "button:has-text('Join from Browser')",
                    "button:has-text('从浏览器加入')",
                    "button:has-text('加入会议')",
                    "a:has-text('Join from Browser')",
                    "[data-action='joinFromBrowser']",
                    ".join-from-browser",
                ]

                for pattern in join_patterns:
                    try:
                        await self._page.click(pattern, timeout=3000)
                        logger.info(f"Clicked join button with pattern: {pattern}")
                        joined = True
                        await asyncio.sleep(2)
                        break
                    except Exception:
                        continue
            except Exception as e:
                logger.debug(f"Method 1 failed: {e}")

        # Method 2: Look for name input directly
        if not joined:
            try:
                name_input = await self._page.wait_for_selector("input[name='uname'], input[placeholder*='name'], input[placeholder*='Name'], input[type='text']", timeout=5000)
                if name_input:
                    await name_input.fill(self.display_name)
                    await asyncio.sleep(0.5)
                    joined = True
            except Exception as e:
                logger.debug(f"Method 2 failed: {e}")

        # Method 3: Fill in password if present
        try:
            password_input = await self._page.wait_for_selector("input[type='password'], input[name='pass']", timeout=3000)
            if password_input and self.meeting_password:
                await password_input.fill(self.meeting_password)
                await asyncio.sleep(0.5)

                # Click submit/join button
                submit_patterns = [
                    "button:has-text('Join')",
                    "button:has-text('加入')",
                    "button[type='submit']",
                    "#joinBtn",
                ]
                for pattern in submit_patterns:
                    try:
                        await self._page.click(pattern, timeout=2000)
                        logger.info("Submitted password")
                        break
                    except Exception:
                        continue
        except Exception:
            pass  # No password required

        # Method 4: Click final join button
        try:
            join_btn_patterns = [
                "button:has-text('Join Meeting')",
                "button:has-text('Join')",
                "button:has-text('加入会议')",
                "#joinBtn",
                "[role='button']:has-text('Join')",
            ]
            for pattern in join_btn_patterns:
                try:
                    await self._page.click(pattern, timeout=2000)
                    logger.info(f"Clicked join button: {pattern}")
                    await asyncio.sleep(3)
                    break
                except Exception:
                    continue
        except Exception:
            pass

        logger.info("Waiting for meeting to load...")
        await asyncio.sleep(5)

        # Enable live captions
        await self._enable_captions()

    async def _enable_captions(self):
        """Enable live captions in the meeting."""
        logger.info("Enabling live captions")

        try:
            # Look for CC button or settings menu
            await asyncio.sleep(3)

            # Try clicking caption toggle buttons
            cc_selectors = [
                "button[aria-label*='closed caption']",
                "button[aria-label*='CC']",
                "button[title*='caption']",
                "[data-testid*='cc']",
                "[data-testid*='caption']",
            ]

            for selector in cc_selectors:
                try:
                    element = await self._page.query_selector(selector)
                    if element:
                        await element.click()
                        logger.info("Enabled live captions via button")
                        return
                except Exception:
                    continue

            # Alternative: open settings and enable captions
            logger.info("Could not find CC button, captions may need manual enablement")

        except Exception as e:
            logger.warning(f"Could not enable captions automatically: {e}")

    async def _monitor_captions(self):
        """Monitor the page for caption updates."""
        logger.info("Monitoring captions")

        last_caption = ""
        caption_count = 0

        while self._is_running:
            try:
                # Try each selector
                for selector in self.CAPTION_SELECTORS:
                    elements = await self._page.query_selector_all(selector)

                    for element in elements:
                        text = await element.inner_text()

                        if not text or not text.strip():
                            continue

                        text = text.strip()

                        # Skip if same as last
                        if text == last_caption:
                            continue

                        # Filter out common false positives
                        text_lower = text.lower()
                        if any(word in text_lower for word in self.FILTER_WORDS):
                            if self.debug:
                                logger.debug(f"Filtered: {text}")
                            continue

                        # Skip very short text (likely noise)
                        if len(text) < 3:
                            if self.debug:
                                logger.debug(f"Too short: {text}")
                            continue

                        # Skip if looks like UI element (contains mostly symbols)
                        if sum(c.isalnum() or c.isspace() for c in text) / len(text) < 0.5:
                            if self.debug:
                                logger.debug(f"Too many symbols: {text}")
                            continue

                        # Valid caption found
                        last_caption = text
                        caption_count += 1

                        # Extract speaker name if available
                        speaker = await self._extract_speaker(element)

                        event = CaptionEvent(text=text, speaker=speaker)
                        logger.info(f"[{caption_count}] Caption: [{speaker or 'Unknown'}] {text}")

                        # Trigger callback
                        if self._on_caption:
                            self._on_caption(event)

                # Wait before next check
                await asyncio.sleep(0.5)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error monitoring captions: {e}")
                await asyncio.sleep(1)

    async def _extract_speaker(self, element) -> Optional[str]:
        """Extract speaker name from caption element."""
        try:
            # Look for speaker name in nearby elements or attributes
            parent = await element.query_selector("..")

            if parent:
                # Check for speaker label
                speaker_selectors = [
                    "[class*='speaker-name']",
                    "[class*='participant-name']",
                    "[data-speaker]",
                ]

                for selector in speaker_selectors:
                    speaker_elem = await parent.query_selector(selector)
                    if speaker_elem:
                        return await speaker_elem.inner_text()

        except Exception:
            pass

        return None

    async def read_captions_for_duration(self, duration: float) -> list[CaptionEvent]:
        """
        Read captions for a specific duration.

        Args:
            duration: Duration in seconds.

        Returns:
            List of caption events.
        """
        captions = []

        def collect_caption(event: CaptionEvent):
            captions.append(event)

        self.on_caption(collect_caption)
        await asyncio.sleep(duration)
        self.on_caption(None)

        return captions


class CaptionsLogger:
    """Simple logger for captions to file or console."""

    def __init__(self, output_file: Optional[str] = None):
        """
        Initialize captions logger.

        Args:
            output_file: Optional file path to log captions.
        """
        self.output_file = output_file
        self._captions: list[CaptionEvent] = []

    def on_caption(self, event: CaptionEvent):
        """Handle a caption event."""
        self._captions.append(event)

        timestamp = event.timestamp.strftime("%H:%M:%S")
        log_line = f"[{timestamp}] [{event.speaker or 'Unknown'}] {event.text}"

        # Print to console
        print(log_line)

        # Write to file if configured
        if self.output_file:
            with open(self.output_file, "a", encoding="utf-8") as f:
                f.write(log_line + "\n")

    def get_all_captions(self) -> list[CaptionEvent]:
        """Get all collected captions."""
        return self._captions

    def clear(self):
        """Clear collected captions."""
        self._captions.clear()


async def test_captions_reader():
    """Test the captions reader."""
    reader = ZoomCaptionsReader(
        meeting_id="123456789",
        display_name="Test Caption Reader",
        headless=False,
    )

    logger = CaptionsLogger(output_file="captions.txt")
    reader.on_caption(logger.on_caption)

    try:
        await reader.start()

        # Run for 5 minutes
        await asyncio.sleep(300)

    except KeyboardInterrupt:
        pass
    finally:
        await reader.stop()

    print(f"\nCollected {len(logger.get_all_captions())} captions")
