"""
WhisperLiveKit Integration

Ultra-low-latency streaming speech-to-text with speaker identification.
"""

import asyncio
import json
from typing import Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import websockets
from loguru import logger


@dataclass
class WLKCaptionEvent:
    """A caption event from WhisperLiveKit."""
    text: str
    speaker: Optional[str] = None
    confidence: float = 0.0
    language: str = "zh"
    timestamp: datetime = None
    is_final: bool = True

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class WhisperLiveKitClient:
    """
    Client for WhisperLiveKit server.

    Connects to a running WLK server via WebSocket for ultra-low-latency transcription.
    """

    def __init__(
        self,
        server_url: str = "ws://localhost:8000/asr",
        language: str = "zh",
        diarization: bool = True,
    ):
        """
        Initialize WhisperLiveKit client.

        Args:
            server_url: WebSocket URL of the WLK server.
            language: Language code (zh, en, auto, etc.).
            diarization: Enable speaker identification.
        """
        self.server_url = server_url
        self.language = language
        self.diarization = diarization

        self._websocket: Optional[websockets.WebSocketClientProtocol] = None
        self._is_connected = False
        self._is_running = False
        self._receive_task: Optional[asyncio.Task] = None

        # Callback for caption events
        self._on_caption: Optional[Callable[[WLKCaptionEvent], None]] = None

        # Audio settings
        self.sample_rate = 16000
        self.channels = 1

    def on_caption(self, callback: Callable[[WLKCaptionEvent], None]):
        """Register a callback for caption events."""
        self._on_caption = callback

    async def start(self):
        """Start the client and connect to the server."""
        logger.info(f"Connecting to WhisperLiveKit server: {self.server_url}")

        try:
            self._websocket = await websockets.connect(
                self.server_url,
                close_timeout=10,
            )
            self._is_connected = True

            logger.info("WebSocket connected, starting receive loop...")

            # Start receiving messages immediately
            self._is_running = True
            self._receive_task = asyncio.create_task(self._receive_loop())

            # Wait for connection to stabilize
            await asyncio.sleep(0.5)

            logger.info("Connected to WhisperLiveKit server, ready to stream audio")

        except Exception as e:
            logger.error(f"Failed to connect to WhisperLiveKit: {e}")
            raise

    async def stop(self):
        """Stop the client and disconnect."""
        logger.info("Stopping WhisperLiveKit client")

        self._is_running = False

        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass

        if self._websocket:
            await self._websocket.close()

        self._is_connected = False
        logger.info("WhisperLiveKit client stopped")

    async def _receive_loop(self):
        """Receive and process messages from the server."""
        message_count = 0
        last_processed_text = ""  # Track last processed text to extract delta only

        while self._is_running:
            try:
                message = await self._websocket.recv()

                if isinstance(message, bytes):
                    # Binary data (audio or other)
                    message_count += 1
                    if message_count <= 2:
                        logger.debug(f"Received binary message #{message_count}: {len(message)} bytes")
                    continue

                # Parse JSON response
                try:
                    data = json.loads(message)
                    # Only log status changes and actual transcriptions
                    status = data.get("status", "")
                    buffer_text = data.get("buffer_transcription", "").strip()

                    # Log important status changes
                    if status == "no_audio_detected":
                        logger.debug(f"[WLK] No audio detected")
                    elif buffer_text:
                        logger.info(f"[WLK Server] Transcribing: {buffer_text[:50]}...")
                    elif status in ["active_transcription", "transcription_complete"]:
                        logger.debug(f"[WLK] Status: {status}")

                    # Handle WLK server response format

                    # Check for transcription in buffer_transcription field
                    buffer_text = data.get("buffer_transcription", "").strip()

                    if buffer_text and buffer_text != last_processed_text:
                        # Extract only new text (delta)
                        if buffer_text.startswith(last_processed_text):
                            new_text = buffer_text[len(last_processed_text):].strip()
                        else:
                            # Text doesn't start with previous (possible reset/correction)
                            new_text = buffer_text

                        if new_text:
                            last_processed_text = buffer_text
                            await self._handle_transcription_text(new_text, data)
                    elif status == "active_transcription":
                        # Check lines array for transcriptions
                        lines = data.get("lines", [])
                        for line in lines:
                            line_text = line.get("text", "").strip()
                            if line_text and line_text != last_processed_text:
                                # Extract only new text (delta)
                                if line_text.startswith(last_processed_text):
                                    new_text = line_text[len(last_processed_text):].strip()
                                else:
                                    new_text = line_text

                                if new_text:
                                    last_processed_text = line_text
                                    speaker_id = line.get("speaker", -2)
                                    await self._handle_transcription_text(new_text, data, speaker_id)
                    elif status in ["no_audio_detected", "active_transcription"]:
                        # Still processing, no new transcription
                        pass

                except json.JSONDecodeError:
                    logger.debug(f"Received non-JSON text: {message[:100]}")

            except websockets.exceptions.ConnectionClosed as e:
                logger.warning(f"WebSocket connection closed: {e}")
                break
            except asyncio.CancelledError:
                logger.debug("Receive loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in receive loop: {e}")
                await asyncio.sleep(1)

    async def _handle_transcription_text(self, text: str, data: dict, speaker_id: int = -2):
        """Handle transcription text from server."""
        if not text or len(text) < 1:
            return

        # Extract speaker if diarization enabled
        speaker = None
        if self.diarization:
            if speaker_id >= 0:
                speaker = f"Speaker {speaker_id}"
            else:
                speaker = f"Speaker {abs(speaker_id)}"

        # Create caption event
        event = WLKCaptionEvent(
            text=text,
            speaker=speaker,
            confidence=data.get("confidence", 0.0),
            language=data.get("language", self.language),
            is_final=True,
        )

        logger.info(f"[WLK] [{speaker or 'Unknown'}] {text}")

        # Trigger callback
        if self._on_caption:
            self._on_caption(event)

    async def _handle_transcription(self, data: dict):
        """Handle transcription message from server."""
        text = data.get("text", "").strip()

        if not text:
            return

        # Extract speaker if diarization enabled
        speaker = None
        if self.diarization:
            speaker = data.get("speaker", f"Speaker {data.get('speaker_id', '?')}")

        # Create caption event
        event = WLKCaptionEvent(
            text=text,
            speaker=speaker,
            confidence=data.get("confidence", 0.0),
            language=data.get("language", self.language),
            is_final=data.get("is_final", True),
        )

        logger.info(f"[WLK] [{speaker or 'Unknown'}] {text}")

        # Trigger callback
        if self._on_caption:
            self._on_caption(event)

    async def send_audio(self, audio_data: np.ndarray):
        """
        Send audio data to the server.

        Args:
            audio_data: Audio data as numpy array (int16 or float32).
        """
        if not self._is_connected:
            logger.warning("Not connected to WhisperLiveKit server")
            return

        try:
            # Ensure audio is float32 and normalized to [-1.0, 1.0]
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            # Clamp to valid range
            audio_data = np.clip(audio_data, -1.0, 1.0)

            # Convert to int16 PCM for WLK
            audio_int16 = (audio_data * 32767).astype(np.int16)

            # Send as bytes
            await self._websocket.send(audio_int16.tobytes())

        except Exception as e:
            logger.error(f"Error sending audio: {e}")


class WhisperLiveKitStreamer:
    """
    Streams audio to WhisperLiveKit and receives real-time transcriptions.

    Combines audio capture with WLK client.
    """

    def __init__(
        self,
        server_url: str = "ws://localhost:8000/asr",
        language: str = "zh",
        diarization: bool = False,
        input_device: Optional[int] = None,
    ):
        """
        Initialize WhisperLiveKit streamer.

        Args:
            server_url: WebSocket URL of the WLK server.
            language: Language code.
            diarization: Enable speaker identification (requires NeMo).
            input_device: Audio input device index (None for default).
        """
        self.server_url = server_url
        self.language = language
        self.diarization = diarization
        self.input_device = input_device

        self._client = WhisperLiveKitClient(
            server_url=server_url,
            language=language,
            diarization=diarization,
        )
        self._audio_capturer = None
        self._is_running = False
        self._stream_task: Optional[asyncio.Task] = None

    def on_caption(self, callback: Callable[[WLKCaptionEvent], None]):
        """Register a callback for caption events."""
        self._client.on_caption(callback)

    async def start(self):
        """Start streaming."""
        logger.info("Starting WhisperLiveKit streamer...")

        # Start audio capture
        from zoom_ai.audio_captions import AudioCapturer

        self._audio_capturer = AudioCapturer(sample_rate=16000, channels=1, device=self.input_device)
        self._audio_capturer.start()

        # Connect to WLK server
        await self._client.start()

        # Start streaming loop
        self._is_running = True
        self._stream_task = asyncio.create_task(self._stream_loop())

        logger.info("WhisperLiveKit streamer started")

    async def stop(self):
        """Stop streaming."""
        logger.info("Stopping WhisperLiveKit streamer")

        self._is_running = False

        if self._stream_task:
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass

        await self._client.stop()

        if self._audio_capturer:
            self._audio_capturer.stop()

        logger.info("WhisperLiveKit streamer stopped")

    async def _stream_loop(self):
        """Stream audio to WLK server."""
        logger.info("Audio streaming loop started")

        chunk_duration_ms = 500  # 500ms chunks
        chunk_count = 0
        sent_count = 0

        while self._is_running:
            try:
                # Get audio chunk
                audio = self._audio_capturer.get_audio_chunk(chunk_duration_ms)

                if audio is not None and len(audio) > 0:
                    chunk_count += 1
                    # Check energy threshold
                    energy = np.mean(np.abs(audio))

                    # Log first few chunks for debugging
                    if chunk_count <= 3:
                        logger.info(f"Audio chunk {chunk_count}: {len(audio)} samples, energy: {energy:.6f}")

                    # Lower threshold for testing (0.00001 instead of 0.0001)
                    if energy > 0.00001:  # Skip silence - very low threshold
                        await self._client.send_audio(audio)
                        sent_count += 1

                        if sent_count <= 3:
                            logger.info(f"Sent audio chunk {sent_count}: energy={energy:.6f}")

                await asyncio.sleep(0.05)  # Small delay

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in stream loop: {e}")
                await asyncio.sleep(1)

        logger.info(f"Audio streaming loop stopped (chunks: {chunk_count}, sent: {sent_count})")


class WLKCaptionLogger:
    """Logger for WhisperLiveKit captions."""

    def __init__(self, output_file: Optional[str] = None):
        """
        Initialize logger.

        Args:
            output_file: Optional file path to log captions.
        """
        self.output_file = output_file
        self._captions: list[WLKCaptionEvent] = []

    def on_caption(self, event: WLKCaptionEvent):
        """Handle a caption event."""
        self._captions.append(event)

        timestamp = event.timestamp.strftime("%H:%M:%S")
        final_marker = "*" if not event.is_final else ""
        log_line = f"[{timestamp}] [{event.speaker or 'Unknown'}]{final_marker} {event.text}"

        # Print to console
        print(log_line)

        # Write to file
        if self.output_file:
            with open(self.output_file, "a", encoding="utf-8") as f:
                f.write(log_line + "\n")

    def get_all_captions(self) -> list[WLKCaptionEvent]:
        """Get all collected captions."""
        return self._captions.copy()

    def clear(self):
        """Clear collected captions."""
        self._captions.clear()


async def test_wlk_captions(
    server_url: str = "ws://localhost:8000/asr",
    language: str = "zh",
    duration: int = 60,
    output_file: Optional[str] = None,
    diarization: bool = False,
):
    """Test WhisperLiveKit captions."""
    logger.info("Testing WhisperLiveKit captions...")
    logger.info(f"Server: {server_url}")
    logger.info(f"Duration: {duration}s")
    logger.info(f"Language: {language}")
    logger.info(f"Diarization: {diarization}")

    streamer = WhisperLiveKitStreamer(
        server_url=server_url,
        language=language,
        diarization=diarization,
    )

    caption_logger = WLKCaptionLogger(output_file=output_file or "logs/wlk_captions.txt")
    streamer.on_caption(caption_logger.on_caption)

    try:
        await streamer.start()
        logger.info(f"Capturing for {duration} seconds...")

        await asyncio.sleep(duration)

        captions = caption_logger.get_all_captions()
        print(f"\nCaptured {len(captions)} captions")

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


class WhisperLiveKitServer:
    """
    Manages a local WhisperLiveKit server process.
    """

    def __init__(
        self,
        model_size: str = "base",
        model_path: Optional[str] = None,
        language: str = "zh",
        diarization: bool = False,
        host: str = "localhost",
        port: int = 8000,
    ):
        """
        Initialize server manager.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large-v3, large-v3-turbo).
            model_path: Custom model path or Hugging Face repo ID (e.g., mlx-community/ivrit-ai-whisper-large-v3-turbo-mlx). Overrides model_size.
            language: Language code.
            diarization: Enable speaker identification (requires NeMo).
            host: Server host.
            port: Server port.
        """
        self.model_size = model_size
        self.model_path = model_path
        self.language = language
        self.diarization = diarization
        self.host = host
        self.port = port

        self._process: Optional[asyncio.subprocess.Process] = None

    async def start(self):
        """Start the WLK server."""
        model = self.model_path or self.model_size
        logger.info(f"Starting WhisperLiveKit server on {self.host}:{self.port}")
        logger.info(f"Model: {model}")
        logger.info(f"Language: {self.language}")
        if self.diarization:
            logger.info(f"Diarization: enabled")

        cmd = [
            "uv",
            "run",
            "wlk",
            "--language", self.language,
            "--host", self.host,
            "--port", str(self.port),
        ]

        # Use --model-path for custom models, --model for built-in sizes
        if self.model_path:
            cmd.extend(["--model-path", self.model_path])
        else:
            cmd.extend(["--model", self.model_size])

        if self.diarization:
            cmd.append("--diarization")

        try:
            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Start a task to log server output
            log_task = asyncio.create_task(self._log_server_output())

            # Wait for server to be ready (with longer timeout for model download)
            # Custom models may need to be downloaded first
            startup_timeout = 120 if self.model_path else 30  # 2min for custom models, 30s for built-in
            logger.info(f"Waiting for WLK server to start (timeout: {startup_timeout}s)...")

            # Wait for server and check if it's still running
            start_time = asyncio.get_event_loop().time()
            while True:
                elapsed = asyncio.get_event_loop().time() - start_time

                # Check if process died
                if self._process.returncode is not None:
                    raise RuntimeError(f"WLK server exited with code {self._process.returncode}")

                # Check timeout with progress logging
                if elapsed > startup_timeout:
                    raise RuntimeError(f"WK server startup timeout after {startup_timeout}s")

                # Log progress every 10s for long waits
                if elapsed > 10 and int(elapsed) % 10 == 0:
                    logger.info(f"Still waiting for WLK server... ({int(elapsed)}/{startup_timeout}s)")
                    if self.model_path and elapsed < 30:
                        logger.info("First time using this model? WLK may be downloading it...")

                # Try to connect to verify server is ready
                try:
                    import websockets
                    test_ws = await asyncio.wait_for(
                        websockets.connect(f"ws://{self.host}:{self.port}/asr", close_timeout=1),
                        timeout=1
                    )
                    await test_ws.close()
                    logger.info("WhisperLiveKit server is ready")
                    break
                except (asyncio.TimeoutError, ConnectionRefusedError, OSError):
                    await asyncio.sleep(1)

        except FileNotFoundError:
            logger.error("wlk command not found. Install with: pip install whisperlivekit")
            raise
        except Exception as e:
            logger.error(f"Failed to start WLK server: {e}")
            raise

    async def _log_server_output(self):
        """Log server output for debugging."""
        async def read_stream(stream, prefix):
            try:
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    line = line.decode().strip()
                    if line:
                        logger.info(f"[WLK Server {prefix}] {line}")
            except Exception as e:
                logger.debug(f"Server {prefix} logging error: {e}")

        try:
            # Read both stdout and stderr concurrently
            await asyncio.gather(
                read_stream(self._process.stdout, "out"),
                read_stream(self._process.stderr, "err"),
            )
        except Exception as e:
            logger.debug(f"Server logging error: {e}")

    async def stop(self):
        """Stop the WLK server."""
        if self._process:
            self._process.terminate()
            await self._process.wait()
            self._process = None
            logger.info("WhisperLiveKit server stopped")

    @property
    def websocket_url(self) -> str:
        """Get the WebSocket URL for the server."""
        return f"ws://{self.host}:{self.port}/asr"


async def test_wlk_with_server(
    model_size: str = "base",
    model_path: Optional[str] = None,
    language: str = "zh",
    duration: int = 60,
    diarization: bool = False,
):
    """Test WLK with auto-started server."""
    server = WhisperLiveKitServer(
        model_size=model_size,
        model_path=model_path,
        language=language,
        diarization=diarization,
    )

    try:
        await server.start()

        streamer = WhisperLiveKitStreamer(
            server_url=server.websocket_url,
            language=language,
            diarization=diarization,
        )

        caption_logger = WLKCaptionLogger(output_file="logs/wlk_captions.txt")
        streamer.on_caption(caption_logger.on_caption)

        await streamer.start()
        logger.info(f"Capturing for {duration} seconds...")

        await asyncio.sleep(duration)

        captions = caption_logger.get_all_captions()
        print(f"\nCaptured {len(captions)} captions")

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
        await server.stop()


if __name__ == "__main__":
    import sys

    # Test with existing server
    if len(sys.argv) > 1 and sys.argv[1] == "--server":
        model = sys.argv[2] if len(sys.argv) > 2 else "base"
        duration = int(sys.argv[3]) if len(sys.argv) > 3 else 60
        asyncio.run(test_wlk_with_server(model_size=model, duration=duration))
    else:
        # Connect to existing server
        url = sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:8000/asr"
        duration = int(sys.argv[2]) if len(sys.argv) > 2 else 60
        asyncio.run(test_wlk_captions(server_url=url, duration=duration))
