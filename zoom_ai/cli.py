"""
Command-line interface for Zoom AI.
"""

import argparse
import asyncio
import sys

from loguru import logger

from zoom_ai.bot import ZoomBot, MultiInstanceBotManager
from zoom_ai.camera import VirtualCamera
from zoom_ai.config import settings, setup_logging


def setup_parser() -> argparse.ArgumentParser:
    """Setup CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="zoom-ai",
        description="Zoom AI Virtual Avatar Meeting Assistant",
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Start command
    start_parser = subparsers.add_parser("start", help="Start the bot")
    start_parser.add_argument(
        "--meeting-id", "-m",
        help="Zoom meeting ID",
    )
    start_parser.add_argument(
        "--meeting-password", "-p",
        help="Zoom meeting password",
    )
    start_parser.add_argument(
        "--bot-name", "-n",
        help="Bot display name",
    )
    start_parser.add_argument(
        "--device", "-d",
        help="Virtual camera device (e.g., /dev/video0)",
    )
    start_parser.add_argument(
        "--instances", "-i",
        type=int,
        default=1,
        help="Number of bot instances to run",
    )

    # Camera test command
    camera_parser = subparsers.add_parser("test-camera", help="Test virtual camera")
    camera_parser.add_argument(
        "--device", "-d",
        default="/dev/video0",
        help="Virtual camera device to test",
    )

    # TTS test command
    tts_parser = subparsers.add_parser("test-tts", help="Test TTS")
    tts_parser.add_argument(
        "--text", "-t",
        default="Hello, this is a test of the Zoom AI text-to-speech system.",
        help="Text to synthesize",
    )

    # Avatar test command
    avatar_parser = subparsers.add_parser("test-avatar", help="Test avatar rendering")
    avatar_parser.add_argument(
        "--duration",
        type=int,
        default=10,
        help="Test duration in seconds",
    )

    return parser


async def cmd_start(args: argparse.Namespace):
    """Handle start command."""
    logger.info("Starting Zoom AI Bot...")

    instances = args.instances or 1

    if instances > 1:
        # Multi-instance mode
        manager = MultiInstanceBotManager(num_instances=instances)
        await manager.run_forever()
    else:
        # Single instance mode
        bot = ZoomBot(
            meeting_id=args.meeting_id,
            meeting_password=args.meeting_password,
            bot_name=args.bot_name,
            device_index=int(args.device.split("video")[-1]) if args.device else None,
        )
        await bot.run_forever()


async def cmd_test_camera(args: argparse.Namespace):
    """Handle test-camera command."""
    logger.info(f"Testing virtual camera: {args.device}")

    # Create test pattern
    import numpy as np

    camera = VirtualCamera(device=args.device)

    if not camera.open():
        logger.error("Failed to open virtual camera")
        return 1

    logger.info("Streaming test pattern for 10 seconds...")

    try:
        for i in range(10 * 30):  # 10 seconds @ 30fps
            # Create test pattern with frame counter
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)

            # Color gradient
            for x in range(1280):
                frame[:, x] = [
                    int(255 * (i % 30) / 30),
                    int(255 * x / 1280),
                    255 - int(255 * x / 1280),
                ]

            # Add text
            import cv2
            cv2.putText(
                frame,
                f"Frame: {i} | Device: {args.device}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

            camera.write(frame)
            await asyncio.sleep(1/30)

        logger.info("Test complete")
        return 0

    except Exception as e:
        logger.error(f"Test failed: {e}")
        return 1
    finally:
        camera.close()


async def cmd_test_tts(args: argparse.Namespace):
    """Handle test-tts command."""
    from zoom_ai.tts import TTSManager

    logger.info("Testing TTS...")
    logger.info(f"Text: {args.text}")

    tts = TTSManager()

    audio_path = await tts.speak(args.text)
    logger.info(f"Audio saved to: {audio_path}")

    # Play audio
    await tts.speak_and_play(args.text)

    return 0


async def cmd_test_avatar(args: argparse.Namespace):
    """Handle test-avatar command."""
    from zoom_ai.avatar import AvatarRendererFactory
    from zoom_ai.camera import VirtualCamera

    logger.info(f"Testing avatar for {args.duration} seconds...")

    camera = VirtualCamera()
    if not camera.open():
        logger.error("Failed to open virtual camera")
        return 1

    avatar = AvatarRendererFactory.create(
        renderer_type=settings.avatar_model,
        width=settings.output_width,
        height=settings.output_height,
        fps=settings.output_fps,
    )

    await avatar.start()

    try:
        # Stream for specified duration
        async for frame in avatar.stream():
            camera.write(frame)

        logger.info("Avatar test complete")
        return 0

    except Exception as e:
        logger.error(f"Avatar test failed: {e}")
        return 1
    finally:
        await avatar.stop()
        camera.close()


def main():
    """Main entry point."""
    parser = setup_parser()
    args = parser.parse_args()

    # Setup logging
    setup_logging()

    # Run command
    if args.command == "start":
        asyncio.run(cmd_start(args))
    elif args.command == "test-camera":
        exit_code = asyncio.run(cmd_test_camera(args))
        sys.exit(exit_code)
    elif args.command == "test-tts":
        exit_code = asyncio.run(cmd_test_tts(args))
        sys.exit(exit_code)
    elif args.command == "test-avatar":
        exit_code = asyncio.run(cmd_test_avatar(args))
        sys.exit(exit_code)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
