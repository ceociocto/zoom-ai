"""
Text-to-Speech (TTS) management.

Supports multiple TTS providers:
- Edge TTS (free, Microsoft Edge)
- Azure Cognitive Services
- ElevenLabs
- Coqui TTS (local, open source)
"""

import asyncio
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from zoom_ai.config import settings


class TTSProvider(ABC):
    """Abstract base class for TTS providers."""

    @abstractmethod
    async def synthesize(self, text: str, output_path: Optional[str] = None) -> str:
        """
        Synthesize speech from text.

        Args:
            text: Text to synthesize.
            output_path: Optional output file path. If None, creates temp file.

        Returns:
            Path to the generated audio file.
        """
        pass

    @abstractmethod
    async def synthesize_stream(self, text: str):
        """
        Synthesize speech as a stream.

        Args:
            text: Text to synthesize.

        Yields:
            Audio chunks as bytes.
        """
        pass


class EdgeTTSProvider(TTSProvider):
    """
    Edge TTS provider using Microsoft Edge's free TTS API.
    No API key required.
    """

    def __init__(self, voice: Optional[str] = None):
        """
        Initialize Edge TTS provider.

        Args:
            voice: Voice name (e.g., "en-US-AriaNeural").
        """
        import edge_tts

        self.voice = voice or settings.tts_voice
        self._communicate = edge_tts.Communicate

    async def synthesize(self, text: str, output_path: Optional[str] = None) -> str:
        """Synthesize speech using Edge TTS."""
        if output_path is None:
            output_path = tempfile.mktemp(suffix=".mp3")

        communicate = self._communicate(text, self.voice)

        await communicate.save(output_path)
        logger.debug(f"Edge TTS synthesized: {text[:50]}... -> {output_path}")
        return output_path

    async def synthesize_stream(self, text: str):
        """Stream synthesized speech."""
        communicate = self._communicate(text, self.voice)

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]


class AzureTTSProvider(TTSProvider):
    """Azure Cognitive Services TTS provider."""

    def __init__(self, key: str, region: str, voice: Optional[str] = None):
        """
        Initialize Azure TTS provider.

        Args:
            key: Azure subscription key.
            region: Azure region (e.g., "eastasia").
            voice: Voice name.
        """
        import azure.cognitiveservices.speech as speechsdk

        self.voice = voice or settings.tts_voice
        self.speech_config = speechsdk.SpeechConfig(
            subscription=key,
            region=region
        )
        self.speech_config.speech_synthesis_voice_name = self.voice

    async def synthesize(self, text: str, output_path: Optional[str] = None) -> str:
        """Synthesize speech using Azure TTS."""
        import azure.cognitiveservices.speech as speechsdk

        if output_path is None:
            output_path = tempfile.mktemp(suffix=".wav")

        # Configure audio output
        audio_config = speechsdk.audio.AudioOutputConfig(filename=output_path)

        # Create synthesizer
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=self.speech_config,
            audio_config=audio_config
        )

        # Synthesize
        result = synthesizer.speak_text_async(text).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            logger.debug(f"Azure TTS synthesized: {text[:50]}... -> {output_path}")
            return output_path
        else:
            raise RuntimeError(f"Azure TTS failed: {result.reason}")

    async def synthesize_stream(self, text: str):
        """Stream synthesized speech using Azure TTS."""
        import azure.cognitiveservices.speech as speechsdk

        # Use push stream for real-time synthesis
        push_stream = speechsdk.audio.PushAudioOutputStream()

        audio_config = speechsdk.audio.AudioOutputConfig(
            stream=push_stream
        )

        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=self.speech_config,
            audio_config=audio_config
        )

        # Start synthesis in background
        def synthesize():
            return synthesizer.speak_text_async(text).get()

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, synthesize)

        # Get audio data
        audio_data = push_stream.get_data()
        yield audio_data


class ElevenLabsTTSProvider(TTSProvider):
    """ElevenLabs TTS provider (high quality)."""

    def __init__(self, api_key: str, voice_id: Optional[str] = None):
        """
        Initialize ElevenLabs TTS provider.

        Args:
            api_key: ElevenLabs API key.
            voice_id: Voice ID (uses default if None).
        """
        self.api_key = api_key
        self.voice_id = voice_id
        self._base_url = "https://api.elevenlabs.io/v1"

    async def synthesize(self, text: str, output_path: Optional[str] = None) -> str:
        """Synthesize speech using ElevenLabs."""
        import aiohttp

        if output_path is None:
            output_path = tempfile.mktemp(suffix=".mp3")

        url = f"{self._base_url}/text-to-speech/{self.voice_id or '21m00Tcm4TlvDq8ikWAM'}"

        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key
        }

        data = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data, headers=headers) as response:
                if response.status == 200:
                    audio_content = await response.read()

                    with open(output_path, 'wb') as f:
                        f.write(audio_content)

                    logger.debug(f"ElevenLabs TTS synthesized: {text[:50]}... -> {output_path}")
                    return output_path
                else:
                    raise RuntimeError(f"ElevenLabs API error: {response.status}")

    async def synthesize_stream(self, text: str):
        """Stream synthesized speech from ElevenLabs."""
        import aiohttp

        url = f"{self._base_url}/text-to-speech/{self.voice_id or '21m00Tcm4TlvDq8ikWAM'}"

        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key
        }

        data = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data, headers=headers) as response:
                if response.status == 200:
                    async for chunk in response.content.iter_chunked(1024):
                        yield chunk
                else:
                    raise RuntimeError(f"ElevenLabs API error: {response.status}")


class TTSManager:
    """
    Manager for TTS operations with automatic provider selection.
    """

    def __init__(self, provider: Optional[str] = None):
        """
        Initialize TTS manager.

        Args:
            provider: TTS provider name ("edge", "azure", "elevenlabs").
        """
        self.provider_name = provider or settings.tts_provider
        self._provider: Optional[TTSProvider] = None

    def _get_provider(self) -> TTSProvider:
        """Get or create TTS provider instance."""
        if self._provider is None:
            if self.provider_name == "edge":
                self._provider = EdgeTTSProvider()
            elif self.provider_name == "azure":
                if not settings.azure_tts_key:
                    raise ValueError("Azure TTS key not configured")
                self._provider = AzureTTSProvider(
                    key=settings.azure_tts_key,
                    region=settings.azure_tts_region
                )
            elif self.provider_name == "elevenlabs":
                if not settings.elevenlabs_api_key:
                    raise ValueError("ElevenLabs API key not configured")
                self._provider = ElevenLabsTTSProvider(
                    api_key=settings.elevenlabs_api_key
                )
            else:
                raise ValueError(f"Unknown TTS provider: {self.provider_name}")

        return self._provider

    async def speak(self, text: str, output_path: Optional[str] = None) -> str:
        """
        Synthesize speech from text.

        Args:
            text: Text to speak.
            output_path: Optional output file path.

        Returns:
            Path to the generated audio file.
        """
        provider = self._get_provider()
        return await provider.synthesize(text, output_path)

    async def speak_stream(self, text: str):
        """
        Stream synthesized speech.

        Args:
            text: Text to speak.

        Yields:
            Audio chunks as bytes.
        """
        provider = self._get_provider()
        async for chunk in provider.synthesize_stream(text):
            yield chunk

    async def speak_and_play(self, text: str):
        """
        Synthesize and play speech using system audio.

        Args:
            text: Text to speak.
        """
        import subprocess

        # Synthesize to temp file
        audio_path = await self.speak(text)

        # Play using ffplay
        process = await asyncio.create_subprocess_exec(
            "ffplay",
            "-nodisp",
            "-autoexit",
            "-loglevel", "quiet",
            audio_path
        )

        await process.wait()

        # Clean up temp file
        Path(audio_path).unlink(missing_ok=True)
