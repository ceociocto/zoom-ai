"""
Simple test for Chinese text on virtual camera.
"""
import asyncio
import numpy as np
import cv2
from PIL import ImageFont, ImageDraw, Image

from zoom_ai.camera import VirtualCamera


async def test_chinese_text():
    """Test Chinese text rendering on virtual camera."""
    camera = VirtualCamera(width=1280, height=720, fps=30)

    if not camera.open():
        print("Failed to open camera")
        return

    print(f"Camera opened: {camera.device}")
    print("Testing Chinese text for 10 seconds...")

    # Load Chinese font
    font = ImageFont.truetype("/System/Library/Fonts/STHeiti Light.ttc", 50)
    print("Chinese font loaded!")

    for i in range(300):  # 10 seconds
        # Create frame
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[:, :] = (50, 100, 150)  # Blue background

        # Add English title with OpenCV
        cv2.putText(frame, "Chinese Text Test", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Add Chinese text with PIL
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)

        test_texts = [
            "你好世界 Hello World",
            "测试中文字符显示",
            "这是语音识别的字幕",
        ]

        y = 150
        for text in test_texts:
            draw.text((50, y), text, font=font, fill=(255, 255, 0))
            y += 60

        # Add frame counter
        draw.text((50, 400), f"Frame: {i}/300", font=font, fill=(200, 200, 200))

        # Convert back
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        camera.write(frame)
        await asyncio.sleep(1/30)

    camera.close()
    print("Test complete!")


if __name__ == "__main__":
    asyncio.run(test_chinese_text())
