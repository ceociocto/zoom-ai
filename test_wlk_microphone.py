"""
Test WLK with real microphone input.
"""
import asyncio
import numpy as np
import websockets
import json
from loguru import logger


async def test_wlk_with_microphone():
    """Test WLK with real microphone input."""
    try:
        import sounddevice as sd
    except ImportError:
        logger.error("sounddevice not installed. Run: pip install sounddevice")
        return

    server_url = "ws://localhost:8000/asr"
    sample_rate = 16000
    channels = 1

    logger.info(f"Connecting to {server_url}...")
    logger.info("🎤 Please speak into your microphone!")

    # Audio queue
    audio_queue = asyncio.Queue()

    def audio_callback(indata, frames, time, status):
        """Sounddevice audio callback."""
        if status:
            logger.warning(f"Audio status: {status}")
        # Put audio data in queue
        audio_queue.put_nowait(indata[:, 0].copy())

    # Start audio capture
    stream = sd.InputStream(
        samplerate=sample_rate,
        channels=channels,
        dtype=np.float32,
        callback=audio_callback,
        blocksize=1600,  # 100ms blocks
    )

    stream.start()
    logger.info("✅ Microphone opened")

    try:
        async with websockets.connect(server_url) as ws:
            logger.info("✅ Connected to WLK server")

            # Receive task
            received_captions = []

            async def receive_messages():
                while True:
                    try:
                        message = await ws.recv()
                        if isinstance(message, str):
                            data = json.loads(message)
                            transcription = data.get("buffer_transcription", "").strip()
                            if transcription:
                                logger.success(f"📝 {transcription}")
                                received_captions.append(transcription)
                    except Exception as e:
                        logger.error(f"Receive error: {e}")
                        break

            receive_task = asyncio.create_task(receive_messages())

            # Stream audio for 30 seconds
            logger.info("📡 Streaming audio for 30 seconds...")
            logger.info("▶️  Start speaking now!")

            for i in range(300):  # 30 seconds
                try:
                    # Get audio chunk (100ms timeout)
                    audio = await asyncio.wait_for(audio_queue.get(), timeout=0.1)

                    # Convert to int16 PCM
                    audio = np.clip(audio, -1.0, 1.0)
                    audio_int16 = (audio * 32767).astype(np.int16)

                    # Send to server
                    await ws.send(audio_int16.tobytes())

                    # Log progress every 5 seconds
                    if i % 50 == 0:
                        logger.info(f"⏱️  {i//10}s elapsed...")

                except asyncio.TimeoutError:
                    # No audio available, send silence
                    silence = np.zeros(1600, dtype=np.int16)
                    await ws.send(silence.tobytes())

            receive_task.cancel()

            logger.info(f"\n{'='*60}")
            logger.info(f"Test complete! Captions received: {len(received_captions)}")
            if received_captions:
                logger.info("Captions:")
                for cap in received_captions:
                    logger.info(f"  - {cap}")
            else:
                logger.warning("No captions received. Please check:")
                logger.info("  1. You spoke loudly enough")
                logger.info("  2. Microphone is working")
                logger.info("  3. WLK server has the correct language model")
            logger.info(f"{'='*60}\n")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        stream.stop()
        logger.info("Microphone closed")


if __name__ == "__main__":
    asyncio.run(test_wlk_with_microphone())
