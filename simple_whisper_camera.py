"""
Simple Whisper + Camera test with Chinese text support.
"""
import asyncio
import numpy as np
import cv2
from PIL import ImageFont, ImageDraw, Image

from zoom_ai.audio_captions import AudioCaptionReader
from zoom_ai.camera import VirtualCamera


async def simple_test():
    """Simple test with guaranteed Chinese text display."""
    # Setup
    camera = VirtualCamera(width=1280, height=720, fps=30)
    transcriber = AudioCaptionReader(model_size="base", language="zh")

    # Load Chinese font
    font = ImageFont.truetype("/System/Library/Fonts/STHeiti Light.ttc", 45)
    print("✅ Chinese font loaded!")

    # Captions storage
    captions = []

    def on_caption(event):
        captions.append(event.text)
        print(f"📺 Caption: {event.text[:30]}...")

    transcriber.on_caption(on_caption)

    # Start
    await transcriber.start()
    if not camera.open():
        print("❌ Camera failed")
        return
    print(f"✅ Camera: {camera.device}")

    print("\n" + "="*60)
    print("Speak now! Captions will appear on virtual camera.")
    print("Running for 15 seconds...")
    print("="*60 + "\n")

    # Run loop
    for i in range(450):  # 15 seconds @ 30fps
        # Create background
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[:, :] = (40, 80, 120)

        # Title
        cv2.putText(frame, "Whisper Captions", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Captions with PIL
        if captions:
            y_start = 720 - len(captions) * 55 - 20

            # Background
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, y_start), (1280, 720), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

            # Text with PIL
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img)

            y = y_start + 50
            for cap in list(captions)[-5:]:  # Last 5
                draw.text((30, y - 35), cap, font=font, fill=(255, 255, 0))
                y += 55

            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        camera.write(frame)
        await asyncio.sleep(1/30)

    # Cleanup
    await transcriber.stop()
    camera.close()

    print(f"\n✅ Done! Captured {len(captions)} captions")


if __name__ == "__main__":
    asyncio.run(simple_test())
