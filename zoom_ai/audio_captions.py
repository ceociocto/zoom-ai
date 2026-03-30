"""
Audio-based Caption Reader using Whisper

Captures system audio and transcribes using OpenAI Whisper.
"""

import asyncio
import queue
import threading
from typing import Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from loguru import logger


@dataclass
class AudioCaptionEvent:
    """A caption event from audio transcription."""
    text: str
    confidence: float = 0.0
    timestamp: datetime = None
    language: str = "zh"

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class AudioCapturer:
    """Captures system audio using sounddevice or pyaudio."""

    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        """
        Initialize audio capturer.

        Args:
            sample_rate: Audio sample rate (16kHz for Whisper).
            channels: Number of audio channels (1=mono).
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.is_recording = False
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self.stream = None

    def start(self):
        """Start audio capture."""
        try:
            import sounddevice as sd

            self.is_recording = True

            def audio_callback(indata, frames, time, status):
                """Callback for audio stream."""
                if status:
                    logger.warning(f"Audio callback status: {status}")

                # sounddevice returns float32 by default, convert to float32
                audio_data = indata[:, 0].copy().astype(np.float32)
                self.audio_queue.put(audio_data)

            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32,  # Explicitly set dtype
                callback=audio_callback,
                blocksize=1600,  # 100ms blocks
            )

            self.stream.start()
            logger.info(f"Audio capture started: {self.sample_rate}Hz")

        except ImportError:
            # Fallback to pyaudio
            self._start_pyaudio()

    def _start_pyaudio(self):
        """Start with pyaudio fallback."""
        import pyaudio

        self.is_recording = True

        p = pyaudio.PyAudio()

        def audio_callback(in_data, frame_count, time_info, status):
            """Callback for pyaudio stream."""
            audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
            self.audio_queue.put(audio_data)
            return (in_data, pyaudio.paContinue)

        self.stream = p.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            stream_callback=audio_callback,
            frames_per_buffer=1600,
        )

        self.stream.start_stream()
        logger.info(f"Audio capture started (pyaudio): {self.sample_rate}Hz")

    def stop(self):
        """Stop audio capture."""
        self.is_recording = False

        if self.stream:
            if hasattr(self.stream, 'stop'):
                self.stream.stop()
            if hasattr(self.stream, 'close'):
                self.stream.close()

        logger.info("Audio capture stopped")

    def get_audio_chunk(self, duration_ms: int = 5000) -> Optional[np.ndarray]:
        """
        Get audio chunk of specified duration.

        Args:
            duration_ms: Duration in milliseconds.

        Returns:
            Audio data as numpy array or None.
        """
        samples_needed = int(self.sample_rate * duration_ms / 1000)
        chunks = []

        while len(chunks) * 1600 < samples_needed:
            try:
                chunk = self.audio_queue.get(timeout=0.1)
                chunks.append(chunk)
            except queue.Empty:
                if not self.is_recording:
                    break

        if not chunks:
            return None

        return np.concatenate(chunks)[:samples_needed]


class WhisperTranscriber:
    """Transcribes audio using OpenAI Whisper."""

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        language: str = "zh",
    ):
        """
        Initialize Whisper transcriber.

        Args:
            model_size: Model size (tiny, base, small, medium, large).
            device: Device to run on (cpu, cuda, mps).
            language: Language code (zh, en, etc.).
        """
        self.model_size = model_size
        self.device = device
        self.language = language
        self.model = None

    def load_model(self):
        """Load Whisper model."""
        import whisper

        logger.info(f"Loading Whisper model: {self.model_size}")

        self.model = whisper.load_model(
            self.model_size,
            device=self.device,
        )

        logger.info("Whisper model loaded")

    def transcribe(self, audio: np.ndarray) -> AudioCaptionEvent:
        """
        Transcribe audio array.

        Args:
            audio: Audio data as numpy array (float32, normalized).

        Returns:
            Caption event with transcribed text.
        """
        if self.model is None:
            self.load_model()

        # Ensure audio is float32 and normalized
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Normalize if needed
        if np.abs(audio).max() > 1.0:
            audio = audio / 32768.0

        # Transcribe
        result = self.model.transcribe(
            audio,
            language=self.language,
            fp16=False,  # Use fp32 for compatibility
        )

        # Get full text
        text = result.get("text", "").strip()

        # Calculate average confidence
        segments = result.get("segments", [])
        confidence = 0.0
        if segments:
            confidence = sum(s.get("avg_logprob", 0) for s in segments) / len(segments)
            # Convert logprob to confidence (rough approximation)
            confidence = max(0, min(1, (confidence + 2) / 4))

        return AudioCaptionEvent(
            text=text,
            confidence=confidence,
            language=self.language,
        )


class AudioCaptionReader:
    """
    Reads captions by capturing audio and transcribing with Whisper.

    This is an alternative to DOM-based caption reading.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        language: str = "zh",
        chunk_duration_ms: int = 5000,
    ):
        """
        Initialize audio caption reader.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large).
            device: Device to run on (cpu, cuda, mps).
            language: Language code (zh, en, auto for auto-detect).
            chunk_duration_ms: Audio chunk duration in milliseconds.
        """
        self.chunk_duration_ms = chunk_duration_ms
        self.language = language

        self._capturer = AudioCapturer(sample_rate=16000, channels=1)
        self._transcriber = WhisperTranscriber(
            model_size=model_size,
            device=device,
            language=None if language == "auto" else language,
        )

        self._is_running = False
        self._transcribe_task: Optional[asyncio.Task] = None
        self._on_caption: Optional[Callable[[AudioCaptionEvent], None]] = None

    def on_caption(self, callback: Callable[[AudioCaptionEvent], None]):
        """Register a callback for caption events."""
        self._on_caption = callback

    async def start(self):
        """Start the audio caption reader."""
        logger.info("Starting audio caption reader...")

        # Load Whisper model
        self._transcriber.load_model()

        # Start audio capture
        self._capturer.start()

        # Start transcription loop
        self._is_running = True
        self._transcribe_task = asyncio.create_task(self._transcribe_loop())

        logger.info("Audio caption reader started")

    async def stop(self):
        """Stop the audio caption reader."""
        logger.info("Stopping audio caption reader...")

        self._is_running = False

        if self._transcribe_task:
            self._transcribe_task.cancel()
            try:
                await self._transcribe_task
            except asyncio.CancelledError:
                pass

        self._capturer.stop()

        logger.info("Audio caption reader stopped")

    async def _transcribe_loop(self):
        """Main transcription loop."""
        logger.info("Transcription loop started")

        while self._is_running:
            try:
                # Get audio chunk
                audio = self._capturer.get_audio_chunk(self.chunk_duration_ms)

                if audio is None or len(audio) < 1000:
                    await asyncio.sleep(0.1)
                    continue

                # Check if audio has content (energy threshold)
                energy = np.mean(np.abs(audio))
                if energy < 0.001:  # Silence threshold
                    if energy > 0:
                        logger.debug(f"Audio too quiet: {energy}")
                    await asyncio.sleep(0.1)
                    continue

                logger.debug(f"Transcribing audio chunk: {len(audio)} samples, energy: {energy}")

                # Run transcription in thread to avoid blocking
                loop = asyncio.get_event_loop()

                def transcribe():
                    return self._transcriber.transcribe(audio)

                event = await loop.run_in_executor(None, transcribe)

                # Filter empty or very short results
                if event.text and len(event.text) > 1:
                    logger.info(f"Transcription: {event.text}")

                    # Trigger callback
                    if self._on_caption:
                        logger.debug(f"Triggering callback with: {event.text[:30]}...")
                        self._on_caption(event)
                    else:
                        logger.warning("No callback registered!")
                else:
                    logger.debug("No transcription result")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in transcription loop: {e}")
                await asyncio.sleep(1)

        logger.info("Transcription loop stopped")


class AudioCaptionLogger:
    """Logger for audio captions."""

    def __init__(self, output_file: Optional[str] = None):
        """
        Initialize logger.

        Args:
            output_file: Optional file path to log captions.
        """
        self.output_file = output_file
        self._captions: list[AudioCaptionEvent] = []

    def on_caption(self, event: AudioCaptionEvent):
        """Handle a caption event."""
        self._captions.append(event)

        timestamp = event.timestamp.strftime("%H:%M:%S")
        conf_pct = int(event.confidence * 100) if event.confidence > 0 else "?"
        log_line = f"[{timestamp}] [{event.language}] ({conf_pct}%) {event.text}"

        # Print to console
        print(log_line)

        # Write to file
        if self.output_file:
            with open(self.output_file, "a", encoding="utf-8") as f:
                f.write(log_line + "\n")

    def get_all_captions(self) -> list[AudioCaptionEvent]:
        """Get all collected captions."""
        return self._captions.copy()

    def clear(self):
        """Clear collected captions."""
        self._captions.clear()


async def test_audio_captions(
    model_size: str = "base",
    duration: int = 60,
    output_file: Optional[str] = None,
    language: str = "zh",
):
    """Test audio caption reader."""
    logger.info("Testing audio caption reader...")
    logger.info(f"Model: {model_size}, Duration: {duration}s, Language: {language}")

    reader = AudioCaptionReader(
        model_size=model_size,
        language=language,
    )

    logger = AudioCaptionLogger(output_file=output_file or "logs/audio_captions.txt")
    reader.on_caption(logger.on_caption)

    try:
        await reader.start()
        logger.info(f"Capturing for {duration} seconds...")
        await asyncio.sleep(duration)

        captions = logger.get_all_captions()
        print(f"\nCaptured {len(captions)} captions")

        return 0

    except KeyboardInterrupt:
        print("\nTest interrupted")
        return 0
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return 1
    finally:
        await reader.stop()


if __name__ == "__main__":
    import sys

    # Get model size from args
    model = sys.argv[1] if len(sys.argv) > 1 else "base"
    duration = int(sys.argv[2]) if len(sys.argv) > 2 else 60

    asyncio.run(test_audio_captions(model_size=model, duration=duration))
