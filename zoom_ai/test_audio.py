"""
Quick test script to verify audio capture is working.
Run this to check if your microphone is being captured correctly.
"""

import asyncio
import numpy as np
from loguru import logger

from zoom_ai.audio_captions import AudioCapturer


async def test_audio_capture(duration: int = 5):
    """Test audio capture."""
    logger.info("Testing audio capture...")

    capturer = AudioCapturer(sample_rate=16000, channels=1)
    capturer.start()

    logger.info(f"Capturing for {duration} seconds...")
    logger.info("Please speak into your microphone!")

    chunk_count = 0
    total_energy = 0
    max_energy = 0

    for i in range(duration * 2):  # Check every 0.5 seconds
        await asyncio.sleep(0.5)
        audio = capturer.get_audio_chunk(500)

        if audio is not None:
            chunk_count += 1
            energy = np.mean(np.abs(audio))
            total_energy += energy
            max_energy = max(max_energy, energy)

            logger.info(f"Chunk {chunk_count}: {len(audio)} samples, energy: {energy:.6f}, max: {max_energy:.6f}")

    capturer.stop()

    avg_energy = total_energy / chunk_count if chunk_count > 0 else 0
    logger.info(f"\nResults:")
    logger.info(f"  Chunks captured: {chunk_count}")
    logger.info(f"  Average energy: {avg_energy:.6f}")
    logger.info(f"  Max energy: {max_energy:.6f}")

    if avg_energy < 0.0001:
        logger.warning("⚠️  Audio energy is very low. Check your microphone!")
    elif avg_energy < 0.001:
        logger.info("⚠️  Audio energy is low. Try speaking louder or closer to the microphone.")
    else:
        logger.info("✅ Audio capture looks good!")


if __name__ == "__main__":
    import sys

    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    asyncio.run(test_audio_capture(duration))
