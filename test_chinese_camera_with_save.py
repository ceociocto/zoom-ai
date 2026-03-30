"""
Test Chinese text and save screenshots.
"""
import asyncio
import numpy as np
import cv2
from PIL import ImageFont, ImageDraw, Image

from zoom_ai.camera import VirtualCamera


async def test_with_screenshot():
    """Test Chinese text and save screenshots."""
    camera = VirtualCamera(width=1280, height=720, fps=30)

    if not camera.open():
        print("Failed to open camera")
        return

    print(f"Camera opened: {camera.device}")

    # Load Chinese font
    font = ImageFont.truetype("/System/Library/Fonts/STHeiti Light.ttc", 50)
    print("Chinese font loaded!")

    # Create test frame and save screenshots
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:, :] = (50, 100, 150)

    # Add Chinese text
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)

    test_texts = [
        ("你好世界", 100),
        ("测试中文字符", 180),
        ("This should work", 260),
    ]

    for text, y in test_texts:
        draw.text((100, y), text, font=font, fill=(255, 255, 0))

    frame_final = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Save screenshot
    cv2.imwrite("/tmp/chinese_camera_test.png", frame_final)
    print("Screenshot saved to: /tmp/chinese_camera_test.png")

    # Stream to camera
    print("Streaming for 5 seconds...")
    for i in range(150):
        camera.write(frame_final)
        await asyncio.sleep(1/30)

    camera.close()
    print("Done! Check the screenshot at /tmp/chinese_camera_test.png")


if __name__ == "__main__":
    asyncio.run(test_with_screenshot())
