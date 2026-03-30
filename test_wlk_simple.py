"""
Simple WLK test to diagnose connection issues.
"""
import asyncio
import numpy as np
import websockets
import json
from loguru import logger


async def test_wlk_connection():
    """Test WLK server connection and audio streaming."""
    server_url = "ws://localhost:8000/asr"

    logger.info(f"Connecting to {server_url}...")

    try:
        async with websockets.connect(server_url) as ws:
            logger.info("✅ Connected to WLK server")

            # Start receive task
            received_messages = []

            async def receive_messages():
                while True:
                    try:
                        message = await ws.recv()
                        received_messages.append(message)
                        logger.info(f"📥 Received: {message[:200] if len(message) > 200 else message}")
                    except Exception as e:
                        logger.error(f"Receive error: {e}")
                        break

            receive_task = asyncio.create_task(receive_messages())

            # Send some test audio (silence + noise)
            logger.info("📤 Sending test audio chunks...")

            for i in range(10):
                # Generate 500ms of audio at 16kHz
                samples = 8000  # 500ms at 16kHz

                # Create test audio (silence with some noise)
                audio = np.random.randn(samples).astype(np.float32) * 0.001
                audio = np.clip(audio, -1.0, 1.0)
                audio_int16 = (audio * 32767).astype(np.int16)

                await ws.send(audio_int16.tobytes())
                logger.info(f"Sent chunk {i+1}/10 ({len(audio_int16)} bytes)")

                await asyncio.sleep(0.5)

            # Wait for responses
            logger.info("Waiting for server responses...")
            await asyncio.sleep(5)

            receive_task.cancel()

            logger.info(f"\n{'='*60}")
            logger.info(f"Total messages received: {len(received_messages)}")
            logger.info(f"{'='*60}")

            if received_messages:
                for msg in received_messages:
                    try:
                        data = json.loads(msg)
                        logger.info(f"Transcription: {data.get('buffer_transcription', data.get('transcription', 'N/A'))}")
                    except:
                        logger.info(f"Raw: {msg[:100]}")
            else:
                logger.warning("⚠️  No transcription received from server")
                logger.info("Please check:")
                logger.info("  1. WLK server is running with correct language model")
                logger.info("  2. Server logs for any errors")
                logger.info("  3. Try speaking into microphone while test runs")

    except Exception as e:
        logger.error(f"❌ Connection failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_wlk_connection())
