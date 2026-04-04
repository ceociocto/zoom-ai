"""
WLK + TTS + Virtual Camera Integration

Features:
- Real-time speech recognition via WhisperLiveKit
- Text-to-speech via GLM API (streaming mode)
- Sentence boundary detection for immediate TTS trigger
- Virtual camera output with captions
"""

import asyncio
import aiohttp
import numpy as np
import cv2
import tempfile
import os
import re
import struct
from typing import Optional, Dict, List, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque

from PIL import ImageFont, ImageDraw, Image
from loguru import logger

from zoom_ai.wlk_captions import WhisperLiveKitStreamer, WLKCaptionEvent
from zoom_ai.camera import VirtualCamera


def _pcm_to_wav(pcm_data: bytes, sample_rate: int = 24000, channels: int = 1, bits_per_sample: int = 16) -> bytes:
    """
    Wrap raw PCM data in a WAV container.

    Args:
        pcm_data: Raw PCM audio data
        sample_rate: Sample rate in Hz (GLM TTS streaming uses 24000Hz)
        channels: Number of audio channels (1=mono)
        bits_per_sample: Bits per sample (16 for PCM16)

    Returns:
        WAV format audio data with headers
    """
    num_frames = len(pcm_data) // (bits_per_sample // 8)
    byte_rate = sample_rate * channels * (bits_per_sample // 8)
    block_align = channels * (bits_per_sample // 8)

    # Build WAV header
    wav_header = struct.pack('<4sI4s', b'RIFF', 36 + len(pcm_data), b'WAVE')
    fmt_chunk = struct.pack('<4sIHHIIHH',
        b'fmt ',           # Chunk ID
        16,                # Chunk size
        1,                 # Audio format (1 = PCM)
        channels,          # Number of channels
        sample_rate,       # Sample rate
        byte_rate,         # Byte rate
        block_align,       # Block align
        bits_per_sample    # Bits per sample
    )
    data_chunk = struct.pack('<4sI', b'data', len(pcm_data))

    return wav_header + fmt_chunk + data_chunk + pcm_data


# Sentence ending patterns - Chinese and English
SENTENCE_END_PATTERN = re.compile(r'[。！？\.!?「」『』【】]')
SENTENCE_PAUSE_PATTERN = re.compile(r'[，、；;]')


@dataclass
class TTSConfig:
    """TTS configuration - supports multiple backends."""
    # Common settings
    model: str = "glm-tts"  # glm-tts, qwen-mlx-tts, etc.
    voice: str = "female"
    speed: float = 1.0

    # GLM TTS API settings (used when model starts with "glm-" or is "glm-tts")
    api_key: str = field(default_factory=lambda: os.getenv("GLM_TTS_API_KEY", ""))
    format: str = "pcm"  # Stream mode only supports PCM
    encode_format: str = "base64"  # or "hex" for stream encoding
    api_url: str = field(default_factory=lambda: os.getenv("GLM_TTS_API_URL", "https://open.bigmodel.cn/api/paas/v4/audio/speech"))

    # MLX TTS settings (used when model contains "mlx" or "qwen")
    mlx_model_path: str = field(default_factory=lambda: os.getenv(
        "QWEN3_TTS_MODEL_PATH",
        "/Volumes/sn7100/jerry/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/downloaded"
    ))
    mlx_sample_rate: int = 24000  # Sample rate for MLX TTS

    # Playback settings
    auto_play: bool = True
    play_command: List[str] = field(default_factory=lambda: ["afplay"])
    use_virtual_audio: bool = False  # Use virtual audio device for Zoom
    virtual_audio_device: Optional[str] = None  # Virtual audio device name

    # Text accumulation for TTS
    min_text_length: int = 2  # Minimum characters before TTS
    max_caption_length: int = 100  # Max characters per caption
    silence_timeout: float = 2.0  # Seconds of silence before treating as sentence end

    @property
    def backend(self) -> str:
        """Determine TTS backend based on model name."""
        model_lower = self.model.lower()
        if model_lower.startswith("glm-") or model_lower == "glm-tts":
            return "glm"
        elif "qwen" in model_lower:
            return "qwen"
        elif "mlx" in model_lower:
            return "mlx"
        else:
            return "glm"  # Default to GLM for backward compatibility


# Backward compatibility alias
GLMTTSConfig = TTSConfig


@dataclass
class CaptionStyle:
    """Caption display style."""
    position: str = "bottom"
    max_lines: int = 5
    line_height: int = 55
    padding: int = 20
    margin_sides: int = 30
    font_size: int = 36
    speaker_font_size: int = 28
    background_color: tuple = (20, 20, 30)
    background_alpha: float = 0.85
    text_color: tuple = (255, 255, 255)
    speaker_colors: List[tuple] = field(default_factory=lambda: [
        (255, 107, 107), (78, 205, 196), (255, 230, 109),
        (133, 193, 233), (255, 121, 198), (162, 155, 254),
    ])
    rounded_corners: int = 15


@dataclass
class CaptionItem:
    """A caption with text and metadata."""
    text: str
    speaker: str
    timestamp: datetime
    is_final: bool = False
    last_update: datetime = None  # Track last time this caption was updated

    def __post_init__(self):
        if self.last_update is None:
            self.last_update = self.timestamp


class ChineseFontLoader:
    """Loader for Chinese fonts."""

    _fonts: Dict[str, ImageFont.FreeTypeFont] = {}

    @classmethod
    def get_font(cls, size: int) -> ImageFont.FreeTypeFont:
        """Get font with specified size."""
        key = f"main_{size}"
        if key not in cls._fonts:
            cls._fonts[key] = cls._load_font(size)
        return cls._fonts[key]

    @classmethod
    def get_speaker_font(cls, size: int) -> ImageFont.FreeTypeFont:
        """Get speaker font."""
        key = f"speaker_{size}"
        if key not in cls._fonts:
            cls._fonts[key] = cls._load_font(size)
        return cls._fonts[key]

    @classmethod
    def _load_font(cls, size: int) -> ImageFont.FreeTypeFont:
        """Load font with fallback chain."""
        font_paths = [
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
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

        logger.warning("Using default font - Chinese may not display correctly")
        return ImageFont.load_default()


class GLMTTSProvider:
    """
    GLM TTS API provider with streaming support.

    Uses streaming mode for faster response:
    - API returns audio chunks via Server-Sent Events (SSE)
    - Chunks are base64/hex encoded PCM data
    - PCM is wrapped in WAV container for playback
    - Sample rate: 24000 Hz (GLM TTS streaming default)
    - Format: 16-bit PCM mono
    """

    def __init__(self, config: TTSConfig):
        """Initialize GLM TTS provider."""
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None
        self._is_playing = False
        self._play_queue: deque = deque()
        self._virtual_player: Optional['VirtualAudioPlayer'] = None  # type: ignore[name-defined]

        # 验证API密钥
        if not self.config.api_key:
            logger.warning("[GLM TTS] GLM_TTS_API_KEY 未设置 - 请在 .env 文件中配置")

        if config.use_virtual_audio:
            # GLM TTS streaming returns 24kHz PCM
            from zoom_ai.virtual_audio import VirtualAudioPlayer
            self._virtual_player = VirtualAudioPlayer(
                sample_rate=24000,
                channels=1,
                device=config.virtual_audio_device
            )
            logger.info(f"Virtual audio player initialized: {self._virtual_player.device} @ 24kHz")

    async def __aenter__(self):
        """Enter context manager."""
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *args):
        """Exit context manager."""
        if self._session:
            await self._session.close()

    async def synthesize(self, text: str) -> Optional[bytes]:
        """
        Synthesize speech from text using GLM TTS API with streaming.

        Args:
            text: Text to synthesize

        Returns:
            Audio data as PCM bytes (16kHz, mono), or None if failed
        """
        if not text or len(text.strip()) < 1:
            return None

        text = text.strip()
        if len(text) < self.config.min_text_length:
            logger.info(f"[TTS: {self.config.model}] Text too short ({len(text)} < {self.config.min_text_length}): {text}")
            return None

        logger.info(f"[TTS: {self.config.model}] 🎙️ 调用 API 合成语音 (流式): {text[:50]}{'...' if len(text) > 50 else ''}")

        try:
            if not self._session:
                self._session = aiohttp.ClientSession()

            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": self.config.model,
                "input": text,
                "voice": self.config.voice,
                "speed": self.config.speed,
                "volume": self.config.volume,
                "response_format": "pcm",  # Stream mode only supports PCM
                "stream": True,  # Enable streaming
                "encode_format": self.config.encode_format,  # base64 or hex
            }

            import base64

            audio_chunks = []
            async with self._session.post(
                self.config.api_url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                logger.info(f"[GLM TTS] Response status: {response.status}, Content-Type: {response.headers.get('Content-Type')}")

                if response.status == 200:
                    # Check if response is SSE or raw binary
                    content_type = response.headers.get('Content-Type', '')
                    logger.info(f"[GLM TTS] Content-Type: {content_type}")

                    # Read all response data first
                    all_data = b""
                    async for chunk in response.content:
                        all_data += chunk

                    logger.info(f"[GLM TTS] Total response size: {len(all_data)} bytes")

                    # Check content type priority
                    if 'audio/' in content_type or 'application/octet-stream' in content_type:
                        # Direct binary response (not SSE)
                        logger.info(f"[GLM TTS] ✅ 收到二进制音频: {len(all_data)} 字节")

                        # If it's already WAV format, return as-is
                        if all_data.startswith(b'RIFF'):
                            return all_data
                        # Otherwise wrap as PCM
                        else:
                            return _pcm_to_wav(all_data, sample_rate=24000)

                    # Parse as SSE (text/event-stream or default)
                    logger.info("[GLM TTS] 解析 SSE 流...")
                    buffer = all_data
                    debug_lines = []
                    raw_response = all_data

                    # Process complete lines
                    while b"\n" in buffer:
                        line_bytes, buffer = buffer.split(b"\n", 1)
                        line = line_bytes.decode('utf-8').strip()

                        # Log first few lines for debugging
                        if len(debug_lines) < 10:
                            debug_lines.append(line[:100])
                            if len(debug_lines) >= 3:
                                logger.debug(f"[GLM TTS] SSE preview: {debug_lines}")

                        # Skip empty lines and SSE control lines
                        if not line or line.startswith(':') or line.startswith('event:'):
                            continue

                        # SSE format: "data: <base64_audio>" or "data: {"choices":[{"delta":{"content":"base64"}}]}"
                        if line.startswith('data:'):
                            data = line[5:].strip()  # Remove "data:" prefix

                            # Check for JSON format
                            if data.startswith('{') and data.endswith('}'):
                                try:
                                    import json
                                    json_data = json.loads(data)
                                    # Handle GLM TTS streaming format: {"choices":[{"delta":{"content":"base64"}}]}
                                    if 'choices' in json_data and len(json_data['choices']) > 0:
                                        delta = json_data['choices'][0].get('delta', {})
                                        data = delta.get('content', '')
                                    else:
                                        # Fallback to direct fields
                                        data = json_data.get('audio', json_data.get('data', ''))
                                except json.JSONDecodeError:
                                    pass

                            if data and data != '[DONE]':
                                try:
                                    # Decode base64/hex encoded audio chunk
                                    if self.config.encode_format == "base64":
                                        # Try standard base64 first, then URL-safe
                                        try:
                                            chunk = base64.b64decode(data)
                                        except Exception:
                                            # Add padding if needed and try URL-safe
                                            padded = data + '=' * (-len(data) % 4)
                                            chunk = base64.urlsafe_b64decode(padded)
                                    else:  # hex
                                        chunk = bytes.fromhex(data)
                                    audio_chunks.append(chunk)
                                except Exception as e:
                                    logger.debug(f"[GLM TTS] Skipping invalid chunk: {e}, data: {data[:50]}")

                    if audio_chunks:
                        pcm_data = b''.join(audio_chunks)
                        # Wrap PCM in WAV for compatibility with audio players
                        audio_data = _pcm_to_wav(pcm_data, sample_rate=24000)
                        logger.info(f"[GLM TTS] ✅ 获得 {len(pcm_data)} 字节 PCM 音频 ({len(audio_chunks)} 个分块)")
                        return audio_data
                    else:
                        raw_preview = raw_response[:500].decode('utf-8', errors='replace')
                        logger.error(f"[GLM TTS] ❌ 未收到音频数据. Raw response preview: {raw_preview}")
                        return None
                else:
                    error_text = await response.text()
                    logger.error(f"[GLM TTS] ❌ API error {response.status}: {error_text}")
                    return None

        except asyncio.TimeoutError:
            logger.error("[GLM TTS] ❌ 请求超时")
            return None
        except Exception as e:
            logger.error(f"[GLM TTS] ❌ 合成失败: {e}")
            return None

    async def play_audio(self, audio_data: bytes) -> bool:
        """
        Play audio data using virtual audio player or system audio player.

        Args:
            audio_data: Audio data as bytes

        Returns:
            True if successful, False otherwise
        """
        if not audio_data:
            return False

        # Use virtual audio player if enabled
        if self._virtual_player:
            logger.info(f"[GLM TTS] 🎧 播放到虚拟音频设备: {self._virtual_player.device}")
            result = await self._virtual_player.play_audio_data(audio_data)
            if result:
                logger.info(f"[GLM TTS] ✅ 虚拟音频播放完成")
            else:
                logger.error(f"[GLM TTS] ❌ 虚拟音频播放失败")
            return result

        # Fallback to system audio player
        try:
            # Write to temp file
            with tempfile.NamedTemporaryFile(
                suffix=f".{self.config.format}",
                delete=False
            ) as f:
                temp_path = f.name
                f.write(audio_data)

            logger.debug(f"[GLM TTS] Playing audio: {temp_path}")

            # Play using system command
            if self.config.play_command[0] == "afplay":
                # macOS
                proc = await asyncio.create_subprocess_exec(
                    "afplay", temp_path
                )
                await proc.wait()
            elif self.config.play_command[0] == "aplay":
                # Linux
                proc = await asyncio.create_subprocess_exec(
                    "aplay", temp_path
                )
                await proc.wait()
            else:
                # Custom command
                proc = await asyncio.create_subprocess_exec(
                    *self.config.play_command, temp_path
                )
                await proc.wait()

            # Clean up temp file
            try:
                os.unlink(temp_path)
            except Exception:
                pass

            logger.info(f"[GLM TTS] 🔊 播放完成")
            return True

        except Exception as e:
            logger.error(f"[GLM TTS] ❌ 播放失败: {e}")
            return False

    async def speak(self, text: str) -> bool:
        """
        Synthesize and speak text.

        Args:
            text: Text to speak

        Returns:
            True if successful, False otherwise
        """
        audio_data = await self.synthesize(text)
        if audio_data:
            logger.info(f"[GLM TTS] 📢 开始播放语音...")
            return await self.play_audio(audio_data)
        logger.warning(f"[GLM TTS] ⚠️ 无音频数据，跳过播放")
        return False


class QwenTTSProvider:
    """
    Qwen3-TTS provider using MLX backend.

    Runs locally on Apple Silicon (M-series) Macs using MLX acceleration.
    Model: Qwen/Qwen3-TTS-12Hz-1.7B-Base
    """

    def __init__(self, config: TTSConfig):
        """Initialize Qwen TTS provider."""
        self.config = config
        self._model = None
        self.sample_rate = 24000
        self._virtual_player: Optional['VirtualAudioPlayer'] = None  # type: ignore[name-defined]

        # Model path from config or env var
        import os
        self._model_path = os.getenv(
            "QWEN3_TTS_MODEL_PATH",
            config.model or "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        )
        logger.info(f"[Qwen TTS] Model path: {self._model_path}")

        if config.use_virtual_audio:
            from zoom_ai.virtual_audio import VirtualAudioPlayer
            self._virtual_player = VirtualAudioPlayer(
                sample_rate=24000,  # Qwen3-TTS uses 24kHz
                channels=1,
                device=config.virtual_audio_device
            )
            logger.info(f"[Qwen TTS] Virtual audio player initialized: {self._virtual_player.device} @ 24kHz")

    async def __aenter__(self):
        """Enter context manager - load Qwen3-TTS model."""
        from mlx_audio.tts.utils import load_model

        logger.info(f"[Qwen TTS] Loading model from: {self._model_path}")

        def load_model_sync():
            self._model = load_model(self._model_path)
            self.sample_rate = self._model.sample_rate
            logger.info(f"[Qwen TTS] Model loaded: {self.sample_rate}Hz")
            logger.info(f"[Qwen TTS] Supported languages: {self._model.get_supported_languages()}")

        # Load model in thread to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, load_model_sync)

        return self

    async def __aexit__(self, *args):
        """Exit context manager - cleanup."""
        import gc
        if self._model is not None:
            del self._model
            self._model = None
        gc.collect()

    async def synthesize(self, text: str) -> Optional[bytes]:
        """
        Synthesize speech from text using Qwen3-TTS.

        Args:
            text: Text to synthesize

        Returns:
            Audio data as WAV bytes (24kHz), or None if failed
        """
        if not text or len(text.strip()) < 1:
            return None

        text = text.strip()
        if len(text) < self.config.min_text_length:
            logger.info(f"[Qwen TTS] Text too short ({len(text)} < {self.config.min_text_length}): {text}")
            return None

        logger.info(f"[Qwen TTS] 🎙️ 合成语音 (本地): {text[:50]}{'...' if len(text) > 50 else ''}")

        try:
            import numpy as np

            def generate_audio():
                # Detect language (simple heuristic)
                lang_code = "chinese" if any('\u4e00' <= c <= '\u9fff' for c in text) else "english"

                # Generate using MLX model
                results = list(self._model.generate(
                    text=text,
                    lang_code=lang_code,
                    verbose=False
                ))

                if results:
                    result = results[0]
                    audio = result.audio  # mx.array
                    audio_np = np.array(audio)
                    return audio_np
                return None

            # Run generation in thread to avoid blocking
            loop = asyncio.get_event_loop()
            audio_array = await loop.run_in_executor(None, generate_audio)

            if audio_array is None:
                logger.error(f"[Qwen TTS] ❌ 生成失败: 无音频输出")
                return None

            logger.info(f"[Qwen TTS] ✅ 生成音频: {len(audio_array)} samples @ {self.sample_rate}Hz")

            # Convert to WAV
            audio_int16 = (audio_array * 32767).astype(np.int16)
            wav_data = _pcm_to_wav(audio_int16.tobytes(), sample_rate=self.sample_rate)

            return wav_data

        except Exception as e:
            logger.error(f"[Qwen TTS] ❌ 合成失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def play_audio(self, audio_data: bytes) -> bool:
        """Play audio data using virtual audio player or system audio player."""
        if not audio_data:
            return False

        if self._virtual_player:
            logger.info(f"[Qwen TTS] 🎧 播放到虚拟音频设备: {self._virtual_player.device}")
            result = await self._virtual_player.play_audio_data(audio_data)
            if result:
                logger.info(f"[Qwen TTS] ✅ 虚拟音频播放完成")
            else:
                logger.error(f"[Qwen TTS] ❌ 虚拟音频播放失败")
            return result

        # Fallback to system audio player
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
                f.write(audio_data)

            logger.debug(f"[Qwen TTS] Playing audio: {temp_path}")

            proc = await asyncio.create_subprocess_exec("afplay", temp_path)
            await proc.wait()

            try:
                os.unlink(temp_path)
            except Exception:
                pass

            logger.info(f"[Qwen TTS] 🔊 播放完成")
            return True

        except Exception as e:
            logger.error(f"[Qwen TTS] ❌ 播放失败: {e}")
            return False

    async def speak(self, text: str) -> bool:
        """Synthesize and speak text."""
        audio_data = await self.synthesize(text)
        if audio_data:
            logger.info(f"[Qwen TTS] 📢 开始播放语音...")
            return await self.play_audio(audio_data)
        logger.warning(f"[Qwen TTS] ⚠️ 无音频数据，跳过播放")
        return False


class MLXTTSProvider:
    """
    MLX-based TTS provider using mlx_audio library.

    Runs locally on Apple Silicon (M-series) Macs with MLX acceleration.
    Supports Qwen3-TTS and other MLX-compatible TTS models.
    """

    def __init__(self, config: TTSConfig):
        """Initialize MLX TTS provider."""
        self.config = config
        self._model = None
        self._virtual_player: Optional['VirtualAudioPlayer'] = None

        if config.use_virtual_audio:
            from zoom_ai.virtual_audio import VirtualAudioPlayer
            self._virtual_player = VirtualAudioPlayer(
                sample_rate=config.mlx_sample_rate,
                channels=1,
                device=config.virtual_audio_device
            )
            logger.info(f"[MLX TTS] Virtual audio player initialized: {self._virtual_player.device} @ {config.mlx_sample_rate}kHz")

    async def __aenter__(self):
        """Enter context manager - load MLX TTS model."""
        from mlx_audio.tts.utils import load_model
        import threading

        logger.info(f"[MLX TTS] Loading model from: {self.config.mlx_model_path}")

        def load_model():
            self._model = load_model(self.config.mlx_model_path)
            logger.info(f"[MLX TTS] Model loaded successfully")
            logger.info(f"[MLX TTS] Supported languages: {self._model.get_supported_languages()}")
            logger.info(f"[MLX TTS] Sample rate: {self._model.sample_rate}Hz")

        # Load model in thread to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, load_model)

        return self

    async def __aexit__(self, *args):
        """Exit context manager - cleanup."""
        if self._model is not None:
            del self._model
            self._model = None

    async def synthesize(self, text: str) -> Optional[bytes]:
        """
        Synthesize speech from text using MLX TTS.

        Args:
            text: Text to synthesize

        Returns:
            Audio data as WAV bytes (24kHz), or None if failed
        """
        if not text or len(text.strip()) < 1:
            return None

        text = text.strip()
        if len(text) < self.config.min_text_length:
            logger.info(f"[MLX TTS] Text too short ({len(text)} < {self.config.min_text_length}): {text}")
            return None

        logger.info(f"[MLX TTS] 🎙️ 合成语音 (本地 MLX): {text[:50]}{'...' if len(text) > 50 else ''}")

        try:
            import numpy as np

            def generate_audio():
                # Generate audio using MLX model
                results = list(self._model.generate(
                    text=text,
                    lang_code='chinese',
                    verbose=False
                ))
                if results:
                    result = results[0]
                    audio = result.audio  # mx.array
                    return np.array(audio)
                return None

            # Run generation in thread to avoid blocking
            loop = asyncio.get_event_loop()
            audio_array = await loop.run_in_executor(None, generate_audio)

            if audio_array is None:
                logger.error("[MLX TTS] ❌ 生成失败")
                return None

            logger.info(f"[MLX TTS] ✅ 生成音频: {len(audio_array)} samples @ {self.config.mlx_sample_rate}Hz")

            # Convert to WAV
            audio_int16 = (audio_array * 32767).astype(np.int16)
            wav_data = _pcm_to_wav(audio_int16.tobytes(), sample_rate=self.config.mlx_sample_rate)

            return wav_data

        except Exception as e:
            logger.error(f"[MLX TTS] ❌ 合成失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def play_audio(self, audio_data: bytes) -> bool:
        """Play audio data using virtual audio player or system audio player."""
        if not audio_data:
            return False

        if self._virtual_player:
            logger.info(f"[MLX TTS] 🎧 播放到虚拟音频设备: {self._virtual_player.device}")
            result = await self._virtual_player.play_audio_data(audio_data)
            if result:
                logger.info(f"[MLX TTS] ✅ 虚拟音频播放完成")
            else:
                logger.error(f"[MLX TTS] ❌ 虚拟音频播放失败")
            return result

        # Fallback to system audio player
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
                f.write(audio_data)

            logger.debug(f"[MLX TTS] Playing audio: {temp_path}")

            proc = await asyncio.create_subprocess_exec("afplay", temp_path)
            await proc.wait()

            try:
                os.unlink(temp_path)
            except Exception:
                pass

            logger.info(f"[MLX TTS] 🔊 播放完成")
            return True

        except Exception as e:
            logger.error(f"[MLX TTS] ❌ 播放失败: {e}")
            return False

    async def speak(self, text: str) -> bool:
        """Synthesize and speak text."""
        audio_data = await self.synthesize(text)
        if audio_data:
            logger.info(f"[MLX TTS] 📢 开始播放语音...")
            return await self.play_audio(audio_data)
        logger.warning(f"[MLX TTS] ⚠️ 无音频数据，跳过播放")
        return False


class TextToSpeech:
    """
    Factory class for TTS providers.

    Automatically selects the appropriate provider based on the model name:
    - "glm-tts" or "glm-*" -> GLMTTSProvider (GLM API)
    - "*mlx*" or "*qwen*" -> MLXTTSProvider (Local MLX)
    - default -> GLMTTSProvider
    """

    def __init__(self, config: TTSConfig):
        """Initialize TTS with appropriate provider."""
        self.config = config
        self._provider: Optional[Union[GLMTTSProvider, QwenTTSProvider, MLXTTSProvider]] = None

    async def __aenter__(self):
        """Enter context manager - initialize selected provider."""
        backend = self.config.backend

        if backend == "qwen":
            logger.info(f"[TTS] Using Qwen backend (transformers) for model: {self.config.model}")
            self._provider = QwenTTSProvider(self.config)
        elif backend == "mlx":
            logger.info(f"[TTS] Using MLX backend for model: {self.config.model}")
            self._provider = MLXTTSProvider(self.config)
        else:  # default to GLM
            logger.info(f"[TTS] Using GLM backend for model: {self.config.model}")
            self._provider = GLMTTSProvider(self.config)

        await self._provider.__aenter__()
        return self

    async def __aexit__(self, *args):
        """Exit context manager - cleanup provider."""
        if self._provider:
            await self._provider.__aexit__(*args)

    async def synthesize(self, text: str) -> Optional[bytes]:
        """Synthesize speech from text."""
        if self._provider:
            return await self._provider.synthesize(text)
        return None

    async def play_audio(self, audio_data: bytes) -> bool:
        """Play audio data."""
        if self._provider:
            return await self._provider.play_audio(audio_data)
        return False

    async def speak(self, text: str) -> bool:
        """Synthesize and speak text."""
        if self._provider:
            return await self._provider.speak(text)
        return False


# Backward compatibility alias
GLMTextToSpeech = GLMTTSProvider


class TTSOverlayRenderer:
    """
    Renderer with TTS integration and sentence boundary detection.

    Features:
    - Accumulates text per speaker
    - Detects sentence boundaries (。！？.!? etc.)
    - Detects silence timeout (2 seconds of no new text)
    - Immediately triggers TTS when sentence ends
    """

    def __init__(self, style: Optional[CaptionStyle] = None, silence_timeout: float = 2.0):
        """Initialize renderer."""
        self.style = style or CaptionStyle()
        self.silence_timeout = silence_timeout
        self._captions: deque = deque(maxlen=self.style.max_lines)
        self._speaker_names: Dict[str, str] = {}
        self._current_caption: Dict[str, CaptionItem] = {}
        self._tts_callback: Optional[Callable[[str, str], None]] = None
        self._finalized_speakers: set = set()  # Track speakers whose captions were finalized by timeout

    def set_tts_callback(self, callback: Callable[[str, str], None]):
        """Set callback for TTS when caption is finalized."""
        self._tts_callback = callback

    def check_silence_timeout(self) -> List[Tuple[str, str]]:
        """
        Check for silence timeout and return list of (speaker_id, text) to finalize.

        Should be called periodically from the main loop.
        Returns list of (speaker_id, final_text) tuples that timed out.
        """
        now = datetime.now()
        timed_out = []

        for speaker_id, caption in list(self._current_caption.items()):
            # Skip if already being processed
            if speaker_id in self._finalized_speakers:
                continue

            # Check time since last update
            elapsed = (now - caption.last_update).total_seconds()

            # Check if timeout exceeded AND has meaningful text
            text_len = len(caption.text.strip())
            if elapsed >= self.silence_timeout and text_len >= 2:  # At least 2 characters
                timed_out.append((speaker_id, caption.text))
                self._finalized_speakers.add(speaker_id)
                logger.info(f"[静音超时] [{self._speaker_names.get(speaker_id, speaker_id)}] "
                           f"静音 {elapsed:.1f}秒，触发TTS: {caption.text[:30]}")

        return timed_out

    def finalize_caption_by_timeout(self, speaker_id: str):
        """
        Finalize a caption due to silence timeout.

        Removes from current captions and adds to finalized list.
        Returns the finalized text, or None if speaker not found.
        """
        if speaker_id in self._current_caption:
            caption = self._current_caption[speaker_id]
            self._captions.append(caption)
            del self._current_caption[speaker_id]
            self._finalized_speakers.discard(speaker_id)
            return caption.text
        return None

    def _detect_sentence_end(self, text: str, new_text: str) -> bool:
        """
        Detect if the new text indicates the end of a sentence.

        Args:
            text: Full accumulated text
            new_text: Newly added text delta

        Returns:
            True if sentence should end here
        """
        # Check if new_text ends with sentence-ending punctuation
        if SENTENCE_END_PATTERN.search(new_text):
            return True

        # Check if text is too long (force split)
        if len(text) >= 100:
            return True

        # Check for pause markers (comma, etc.) - only if text is reasonably long
        if len(text) >= 20 and SENTENCE_PAUSE_PATTERN.search(new_text):
            return True

        return False

    def on_caption(self, event: WLKCaptionEvent):
        """
        Handle caption event with immediate sentence-end TTS trigger.

        This is the core logic:
        1. Accumulate text per speaker
        2. When sentence end detected, finalize caption and trigger TTS immediately
        3. When speaker changes, finalize previous speaker's caption
        4. Update last_update time for silence timeout detection
        """
        speaker_id = event.speaker or "SPEAKER_0"
        now = datetime.now()

        # Speaker is actively talking - remove from timeout tracking
        self._finalized_speakers.discard(speaker_id)

        # Generate speaker name
        if speaker_id not in self._speaker_names:
            speaker_num = len(self._speaker_names) + 1
            self._speaker_names[speaker_id] = f"说话人 {speaker_num}"

        # Check if speaker has active caption
        if speaker_id in self._current_caption:
            current = self._current_caption[speaker_id]
            current.text += event.text
            current.last_update = now  # Update last activity time

            # Check for sentence boundary - immediately trigger TTS if detected
            if self._detect_sentence_end(current.text, event.text):
                # Sentence ended! Finalize and trigger TTS immediately
                final_text = current.text
                self._captions.append(current)
                del self._current_caption[speaker_id]

                logger.info(f"[句子结束] [{self._speaker_names[speaker_id]}] {final_text}")

                # Trigger TTS immediately
                if self._tts_callback:
                    self._tts_callback(final_text, speaker_id)
            else:
                # Still in same sentence - just accumulate (will be re-rendered)
                pass
        else:
            # New caption for this speaker
            caption = CaptionItem(
                text=event.text,
                speaker=speaker_id,
                timestamp=now,
                last_update=now,
            )
            self._current_caption[speaker_id] = caption

        # When different speaker talks, finalize their previous caption
        for sp_id, cap in list(self._current_caption.items()):
            if sp_id != speaker_id:
                self._captions.append(cap)
                del self._current_caption[sp_id]

                final_text = cap.text
                logger.info(f"[说话人切换] [{self._speaker_names[sp_id]}] {final_text}")

                if self._tts_callback:
                    self._tts_callback(final_text, sp_id)

    def _get_display_captions(self) -> List[CaptionItem]:
        """Get all captions to display (finalized + active)."""
        return list(self._captions) + list(self._current_caption.values())

    def render(self, frame: np.ndarray) -> np.ndarray:
        """Render captions on frame."""
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img, 'RGBA')

        width, height = img.size
        captions = self._get_display_captions()

        if not captions:
            return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        line_height = self.style.line_height
        padding = self.style.padding
        margin = self.style.margin_sides

        total_height = len(captions) * line_height + 2 * padding
        y_base = height - total_height - padding - 20

        # Draw background
        bg_color = self.style.background_color
        bg_alpha = int(self.style.background_alpha * 255)

        # Shadow
        self._draw_rounded_rect(
            draw,
            (margin, y_base + 4, width - margin, y_base + 4 + total_height),
            (0, 0, 0, 80),
            self.style.rounded_corners
        )

        # Background card
        self._draw_rounded_rect(
            draw,
            (margin, y_base, width - margin, y_base + total_height),
            (*bg_color, bg_alpha),
            self.style.rounded_corners
        )

        # Draw captions
        font = ChineseFontLoader.get_font(self.style.font_size)
        speaker_font = ChineseFontLoader.get_speaker_font(self.style.speaker_font_size)

        for i, caption in enumerate(captions):
            y = y_base + padding + i * line_height

            speaker_name = self._speaker_names.get(caption.speaker, caption.speaker)
            color = self._get_speaker_color(caption.speaker)

            # Speaker badge
            badge_text = f"{speaker_name}: "
            badge_bbox = draw.textbbox((0, 0), badge_text, font=speaker_font)
            badge_width = badge_bbox[2] - badge_bbox[0]

            self._draw_rounded_rect(
                draw,
                (margin + 10, y + 5, margin + 10 + badge_width, y + line_height - 10),
                (*color, 180),
                8
            )

            draw.text(
                (margin + 15, y + 8),
                badge_text,
                font=speaker_font,
                fill=(255, 255, 255, 255)
            )

            # Caption text
            text_x = margin + 20 + badge_width
            draw.text(
                (text_x, y + 8),
                caption.text,
                font=font,
                fill=(*self.style.text_color, 255)
            )

        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def _draw_rounded_rect(self, draw: ImageDraw.ImageDraw, coords: Tuple, color: Tuple, radius: int):
        """Draw rounded rectangle."""
        x1, y1, x2, y2 = coords
        draw.rounded_rectangle([(x1, y1), (x2, y2)], radius=radius, fill=color)

    def _get_speaker_color(self, speaker_id: str) -> Tuple:
        """Get color for speaker."""
        idx = hash(speaker_id) % len(self.style.speaker_colors)
        return self.style.speaker_colors[idx]

    def clear(self):
        """Clear all captions."""
        self._captions.clear()
        self._speaker_names.clear()
        self._current_caption.clear()


class WLKStreamerWithTTS:
    """
    WLK + TTS + Virtual Camera integration.

    Workflow:
    1. Capture audio from microphone
    2. Send to WLK server for transcription
    3. Detect sentence boundaries
    4. Immediately send completed sentences to GLM TTS
    5. Play synthesized audio
    6. Display captions on virtual camera
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
        tts_config: Optional[GLMTTSConfig] = None,
        caption_style: Optional[CaptionStyle] = None,
        input_device: Optional[int] = None,
    ):
        """Initialize streamer."""
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
            input_device=input_device,
        )

        # TTS
        self._tts_config = tts_config or GLMTTSConfig()
        self._tts: Optional[TextToSpeech] = None

        # Overlay renderer with silence timeout
        self._overlay = TTSOverlayRenderer(
            style=caption_style,
            silence_timeout=self._tts_config.silence_timeout
        )
        self._wlk.on_caption(self._overlay.on_caption)

        # State
        self._is_running = False
        self._stream_task: Optional[asyncio.Task] = None
        self._tts_queue: asyncio.Queue = asyncio.Queue()
        self._tts_task: Optional[asyncio.Task] = None
        self._timeout_check_task: Optional[asyncio.Task] = None
        self._tts_queue: asyncio.Queue = asyncio.Queue()
        self._tts_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the streamer."""
        logger.info("Starting WLK + TTS Streamer...")

        # Open virtual camera
        if not self._camera.open():
            raise RuntimeError("Failed to open virtual camera")

        # Start WLK
        try:
            await self._wlk.start()
        except Exception as e:
            # Clean up camera on failure
            self._camera.close()

            if "Connect call failed" in str(e) or "connection" in str(e).lower():
                logger.error("无法连接到 WLK 服务器!")
                logger.error("请先在另一个终端运行: uv run wlk --model base --language zh --pcm-input")
                raise RuntimeError(
                    "WLK 服务器未运行。请先运行: uv run wlk --model base --language zh --pcm-input"
                ) from e
            raise

        # Initialize TTS
        self._tts = await TextToSpeech(self._tts_config).__aenter__()

        # Set TTS callback - triggers immediately on sentence end
        self._overlay.set_tts_callback(self._on_caption_finalized)

        # Start streaming
        self._is_running = True
        self._stream_task = asyncio.create_task(self._stream_loop())
        self._tts_task = asyncio.create_task(self._tts_loop())
        self._timeout_check_task = asyncio.create_task(self._timeout_check_loop())

        logger.info("WLK + TTS Streamer started")
        logger.info(f"Camera: {self._camera.device} @ {self._camera.width}x{self._camera.height}")
        logger.info(f"TTS Model: {self._tts_config.model}")
        logger.info(f"TTS will trigger on: sentence end (。！？.!?) OR {self._tts_config.silence_timeout}s silence")

    async def stop(self):
        """Stop the streamer."""
        logger.info("Stopping WLK + TTS Streamer...")
        self._is_running = False

        if self._stream_task:
            self._stream_task.cancel()
        if self._tts_task:
            self._tts_task.cancel()
        if self._timeout_check_task:
            self._timeout_check_task.cancel()

        # Only await if task was created
        if self._stream_task:
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass
        if self._tts_task:
            try:
                await self._tts_task
            except asyncio.CancelledError:
                pass
        if self._timeout_check_task:
            try:
                await self._timeout_check_task
            except asyncio.CancelledError:
                pass

        await self._wlk.stop()

        if self._tts:
            await self._tts.__aexit__()

        self._camera.close()
        logger.info("WLK + TTS Streamer stopped")

    def _on_caption_finalized(self, text: str, speaker_id: str):
        """
        Handle finalized caption - queue for TTS immediately.

        This is called as soon as a sentence boundary is detected.
        """
        if self._tts_config.auto_play:
            # Add to TTS queue for immediate playback
            asyncio.create_task(self._tts_queue.put((text, speaker_id)))

    async def _tts_loop(self):
        """TTS playback loop."""
        logger.info("TTS loop started - will play on sentence end")

        while self._is_running:
            try:
                text, speaker_id = await asyncio.wait_for(
                    self._tts_queue.get(),
                    timeout=1.0
                )

                if self._tts:
                    speaker_name = self._overlay._speaker_names.get(speaker_id, speaker_id)
                    logger.info(f"[TTS] [{speaker_name}] 开始播放: {text[:30]}...")
                    await self._tts.speak(text)

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"TTS error: {e}")

        logger.info("TTS loop stopped")

    async def _timeout_check_loop(self):
        """Periodically check for silence timeout and trigger TTS."""
        logger.info(f"Timeout check loop started - checking every {self._tts_config.silence_timeout}s")

        while self._is_running:
            try:
                await asyncio.sleep(self._tts_config.silence_timeout)

                # Check for timed out captions
                timed_out = self._overlay.check_silence_timeout()

                for speaker_id, text in timed_out:
                    # Finalize the caption
                    final_text = self._overlay.finalize_caption_by_timeout(speaker_id)

                    if final_text and self._tts_config.auto_play:
                        # Trigger TTS
                        await self._tts_queue.put((final_text, speaker_id))

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Timeout check error: {e}")

        logger.info("Timeout check loop stopped")

    async def _stream_loop(self):
        """Frame streaming loop."""
        logger.info("Frame streaming loop started")

        while self._is_running:
            try:
                frame = self._generate_frame()

                if frame is not None:
                    frame_with_overlay = self._overlay.render(frame)
                    self._camera.write(frame_with_overlay)

                await asyncio.sleep(1 / self._camera.fps)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Stream loop error: {e}")
                await asyncio.sleep(1)

        logger.info("Frame streaming loop stopped")

    def _generate_frame(self) -> Optional[np.ndarray]:
        """Generate background frame."""
        frame = np.zeros((self._camera.height, self._camera.width, 3), dtype=np.uint8)

        # Gradient background
        for y in range(self._camera.height):
            for x in range(self._camera.width):
                r = int(30 + 50 * x / self._camera.width)
                g = int(20 + 40 * y / self._camera.height)
                b = int(50 + 60 * (x + y) / (self._camera.width + self._camera.height))
                frame[y, x] = (b, g, r)

        # Add title
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)

        font_title = ChineseFontLoader.get_font(40)
        font_sub = ChineseFontLoader.get_font(24)

        draw.text((50, 40), "Zoom AI 实时字幕 + TTS", font=font_title, fill=(255, 255, 255))

        timestamp = datetime.now().strftime("%H:%M:%S")
        tts_status = "启用" if self._tts_config.auto_play else "禁用"
        status_text = f"时间: {timestamp} | TTS: {tts_status}"
        draw.text((50, 100), status_text, font=font_sub, fill=(200, 200, 200))

        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


async def test_wlk_tts(
    wlk_server_url: str = "ws://localhost:8000/asr",
    language: str = "zh",
    diarization: bool = True,
    duration: int = 60,
    auto_tts: bool = True,
    use_virtual_audio: bool = False,
    virtual_audio_device: Optional[str] = None,
    input_device: Optional[int] = None,
    model: str = "glm-tts",
):
    """
    Test WLK + TTS integration with sentence-end detection.

    Args:
        wlk_server_url: WLK server WebSocket URL
        language: Language code
        diarization: Enable speaker identification
        duration: Test duration in seconds
        auto_tts: Enable automatic TTS playback
        use_virtual_audio: Use virtual audio device for Zoom
        virtual_audio_device: Virtual audio device name
        model: TTS model to use (default: glm-tts)
    """
    logger.info("Testing WLK + TTS Integration...")
    logger.info(f"TTS Model: {model}")

    tts_config = GLMTTSConfig(
        auto_play=auto_tts,
        use_virtual_audio=use_virtual_audio,
        virtual_audio_device=virtual_audio_device,
        model=model,
    )

    streamer = WLKStreamerWithTTS(
        wlk_server_url=wlk_server_url,
        language=language,
        diarization=diarization,
        tts_config=tts_config,
        input_device=input_device,
    )

    try:
        await streamer.start()

        print(f"\n{'='*60}")
        print("🎤 WLK + TTS 测试已启动!")
        print(f"📢 请对着麦克风说话")
        print(f"🔊 TTS: {'✅ 启用' if auto_tts else '❌ 禁用'}")
        print(f"🤖 TTS Model: {model}")
        print(f"🎧 虚拟音频: {'✅ 启用' if use_virtual_audio else '❌ 禁用'}")
        print(f"📍 虚拟摄像头: {streamer._camera.device}")
        print(f"⏱️  测试时长: {duration} 秒")
        print(f"⏸️  静音超时: {tts_config.silence_timeout} 秒")
        print(f"{'='*60}")
        print(f"TTS 触发条件:")
        print(f"  1. 句子结束标点 (。！？.!?"")「」『』【】)")
        print(f"  2. 静音超过 {tts_config.silence_timeout} 秒")
        print(f"  3. 文本过长 (>100 字符)")

        if use_virtual_audio:
            print(f"\n🔊 音频输出到虚拟设备: {virtual_audio_device or '自动检测'}")
            print(f"   Zoom 中选择该设备作为麦克风")
        else:
            print(f"\n💡 提示: 使用 --virtual-audio 参数可让 Zoom 听到 TTS 语音")
            print(f"   运行 'uv run python -m zoom_ai.cli test-wlk-tts --setup-audio' 查看设置指南")

        print(f"\n提示: 按 Ctrl+C 提前结束测试\n")

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

    parser = argparse.ArgumentParser(description="WLK + TTS Test")
    parser.add_argument("--server-url", "-s", default="ws://localhost:8000/asr",
                        help="WLK server URL")
    parser.add_argument("--language", "-l", default="zh", help="Language code")
    parser.add_argument("--duration", "-d", type=int, default=60,
                        help="Test duration in seconds")
    parser.add_argument("--no-tts", action="store_true",
                        help="Disable TTS playback")

    args = parser.parse_args()

    asyncio.run(test_wlk_tts(
        wlk_server_url=args.server_url,
        language=args.language,
        duration=args.duration,
        auto_tts=not args.no_tts,
    ))
