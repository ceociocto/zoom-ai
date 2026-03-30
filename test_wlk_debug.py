"""
Debug test for WLK + Virtual Camera integration.
"""
import asyncio
import sys
from datetime import datetime

from loguru import logger

from zoom_ai.wlk_camera_overlay import WLKCameraStreamer, CaptionOverlayRenderer
from zoom_ai.wlk_captions import WLKCaptionLogger


async def test_with_debug():
    """Test with extensive debug output."""

    print("\n" + "="*60)
    print("WLK + Virtual Camera Debug Test")
    print("="*60)

    # Test 1: Check WLK connection
    print("\n[Test 1] Testing WLK connection...")

    from zoom_ai.wlk_captions import WhisperLiveKitStreamer

    caption_count = 0

    def on_caption(event):
        nonlocal caption_count
        caption_count += 1
        print(f"\n✅ CAPTION RECEIVED #{caption_count}:")
        print(f"   Speaker: {event.speaker}")
        print(f"   Text: {event.text}")
        print(f"   Time: {event.timestamp.strftime('%H:%M:%S')}")
        print(f"   Confidence: {event.confidence}")

    wlk = WhisperLiveKitStreamer(
        server_url="ws://localhost:8000/asr",
        language="zh",
        diarization=False,
    )
    wlk.on_caption(on_caption)

    try:
        await wlk.start()
        print("✅ WLK connected successfully")
        print("⏱️  Testing for 15 seconds...")
        print("🎤 Please speak into your microphone now!")

        await asyncio.sleep(15)

        if caption_count > 0:
            print(f"\n✅ SUCCESS: Received {caption_count} captions")
        else:
            print(f"\n⚠️  WARNING: No captions received")
            print("   Check:")
            print("   1. Is your microphone working?")
            print("   2. Is the WLK server running correctly?")
            print("   3. Try running: uv run wlk --model base --language zh")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return
    finally:
        await wlk.stop()

    # Test 2: Virtual camera with overlay
    print("\n" + "="*60)
    print("\n[Test 2] Testing Virtual Camera with Overlay...")
    print("This will stream to 'OBS Virtual Camera' for 30 seconds")
    print("Open OBS or any video app to see the output\n")

    streamer = WLKCameraStreamer(
        wlk_server_url="ws://localhost:8000/asr",
        language="zh",
        diarization=False,
    )

    # Add debug to overlay
    original_on_caption = streamer._overlay.on_caption

    def debug_on_caption(event):
        print(f"\n📺 OVERLAY: [{event.speaker or 'Unknown'}] {event.text}")
        original_on_caption(event)

    streamer._overlay.on_caption = debug_on_caption

    try:
        await streamer.start()
        print("✅ Virtual camera streaming started")
        print("📷 Device:", streamer._camera.device)
        print("🎤 Speak now to see captions on the virtual camera!")
        print("⏱️  Running for 30 seconds...\n")

        await asyncio.sleep(30)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await streamer.stop()

    print("\n" + "="*60)
    print("Test complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    try:
        asyncio.run(test_with_debug())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
