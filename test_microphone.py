"""
Microphone level monitor - Test if your microphone is working.
"""
import asyncio
import numpy as np
from zoom_ai.audio_captions import AudioCapturer

async def test_microphone():
    """Test microphone input levels."""
    print("\n" + "="*60)
    print("Microphone Level Test")
    print("="*60)
    print("\n🎤 Speak into your microphone...")
    print("📊 You should see energy levels above 0.01 when speaking")
    print("🛑 Press Ctrl+C to stop\n")

    capturer = AudioCapturer(sample_rate=16000, channels=1)
    capturer.start()

    try:
        print("Time    | Energy Level | Status")
        print("-" * 45)

        for i in range(100):  # Test for ~10 seconds
            audio = capturer.get_audio_chunk(100)  # 100ms chunks

            if audio is not None:
                energy = np.mean(np.abs(audio))

                # Determine status
                if energy > 0.05:
                    status = "🔊 LOUD SPEECH"
                elif energy > 0.01:
                    status = "🗣️  SPEECH DETECTED"
                elif energy > 0.001:
                    status = "🤫 Quiet"
                else:
                    status = "⚠️  Too Quiet / No Audio"

                print(f"{i:3d}s    | {energy:.6f}     | {status}")
            else:
                print(f"{i:3d}s    | ---          | ❌ No audio data")

            await asyncio.sleep(1)

    except KeyboardInterrupt:
        print("\n\nTest stopped by user")
    finally:
        capturer.stop()
        print("\n" + "="*60)
        print("Test complete!")
        print("="*60)

if __name__ == "__main__":
    asyncio.run(test_microphone())
