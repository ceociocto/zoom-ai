#!/usr/bin/env python3
"""
Qwen3-TTS test script using MLX backend.
"""

import asyncio
import argparse
import os
import wave
import numpy as np
from pathlib import Path
from loguru import logger


# Qwen3-TTS model path (MLX backend) - can be overridden by QWEN3_TTS_MODEL_PATH env var
QWEN3_TTS_PATH = os.getenv(
    "QWEN3_TTS_MODEL_PATH",
    "/Volumes/sn7100/jerry/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/downloaded"
)


def _pcm_to_wav(pcm_data: bytes, sample_rate: int = 24000, channels: int = 1, bits_per_sample: int = 16) -> bytes:
    """Wrap raw PCM data in a WAV container."""
    import struct
    num_frames = len(pcm_data) // (bits_per_sample // 8)
    byte_rate = sample_rate * channels * (bits_per_sample // 8)
    block_align = channels * (bits_per_sample // 8)

    wav_header = struct.pack('<4sI4s', b'RIFF', 36 + len(pcm_data), b'WAVE')
    fmt_chunk = struct.pack('<4sIHHIIHH',
        b'fmt ', 16, 1, channels, sample_rate, byte_rate, block_align, bits_per_sample
    )
    data_chunk = struct.pack('<4sI', b'data', len(pcm_data))

    return wav_header + fmt_chunk + data_chunk + pcm_data


class Qwen3TTSProvider:
    """Qwen3-TTS provider using MLX backend."""

    def __init__(self, model_path: str = QWEN3_TTS_PATH):
        from mlx_audio.tts.utils import load_model
        logger.info(f"[Qwen3-TTS] Loading model from: {model_path}")
        self._model = load_model(model_path)
        self.sample_rate = self._model.sample_rate
        logger.info(f"[Qwen3-TTS] Model loaded: {self.sample_rate}Hz")
        logger.info(f"[Qwen3-TTS] Supported languages: {self._model.get_supported_languages()}")

    def synthesize(self, text: str, lang_code: str = "chinese") -> bytes:
        """Synthesize speech from text."""
        logger.info(f"[Qwen3-TTS] Synthesizing: {text[:50]}...")

        results = list(self._model.generate(
            text=text,
            lang_code=lang_code,
            verbose=False
        ))

        if results:
            result = results[0]
            audio = result.audio  # mx.array
            audio_np = np.array(audio)
            audio_int16 = (audio_np * 32767).astype(np.int16)
            wav_data = _pcm_to_wav(audio_int16.tobytes(), sample_rate=self.sample_rate)
            logger.info(f"[Qwen3-TTS] Generated {len(audio_np)} samples ({result.audio_duration})")
            return wav_data
        return None

    def save(self, text: str, output_path: str, lang_code: str = "chinese") -> bool:
        """Synthesize and save to WAV file."""
        wav_data = self.synthesize(text, lang_code)
        if wav_data:
            with open(output_path, 'wb') as f:
                f.write(wav_data)
            logger.info(f"[Qwen3-TTS] Saved to: {output_path}")
            return True
        return False


async def test_qwen3_tts(args):
    """Test Qwen3-TTS."""
    text = args.text or "你好，这是 Qwen3-TTS 语音合成测试。"
    output = args.output or "/tmp/qwen3_tts_test.wav"

    logger.info("=" * 50)
    logger.info("Qwen3-TTS 测试")
    logger.info("=" * 50)
    logger.info(f"文本: {text}")
    logger.info(f"输出: {output}")
    logger.info(f"语言: {args.lang}")

    tts = Qwen3TTSProvider(args.model)

    if args.play:
        # Synthesize and play
        import subprocess
        wav_data = tts.synthesize(text, args.lang)
        if wav_data:
            # Save to temp file
            temp_path = "/tmp/qwen3_tts_temp.wav"
            with open(temp_path, 'wb') as f:
                f.write(wav_data)

            # Play
            logger.info(f"[Qwen3-TTS] Playing audio...")
            proc = await asyncio.create_subprocess_exec("afplay", temp_path)
            await proc.wait()

            # Cleanup
            import os
            try:
                os.unlink(temp_path)
            except:
                pass
    else:
        # Just save
        tts.save(text, output, args.lang)

    logger.info("=" * 50)
    logger.info("✓ 测试完成")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS Test (MLX Backend)")
    parser.add_argument("--text", "-t", help="Text to synthesize")
    parser.add_argument("--output", "-o", help="Output WAV file path")
    parser.add_argument("--play", "-p", action="store_true", help="Play audio after synthesis")
    parser.add_argument("--lang", "-l", default="chinese", help="Language code")
    parser.add_argument(
        "--model", "-m",
        default=QWEN3_TTS_PATH,
        help=f"Model path (default: {QWEN3_TTS_PATH})"
    )
    args = parser.parse_args()

    return asyncio.run(test_qwen3_tts(args))


if __name__ == "__main__":
    import sys
    sys.exit(main())
