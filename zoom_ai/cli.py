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

    # Audio captions test command (Whisper)
    audio_captions_parser = subparsers.add_parser("test-audio-captions", help="Test audio captions with Whisper")
    audio_captions_parser.add_argument(
        "--model", "-m",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)",
    )
    audio_captions_parser.add_argument(
        "--duration", "-d",
        type=int,
        default=60,
        help="Duration to capture captions (seconds)",
    )
    audio_captions_parser.add_argument(
        "--language", "-l",
        default="zh",
        help="Language code (zh, en, auto, etc.)",
    )
    audio_captions_parser.add_argument(
        "--output", "-o",
        help="Output file for captions",
    )

    # List audio devices command
    list_audio_parser = subparsers.add_parser("list-audio-devices", help="List available audio input devices")

    # WhisperLiveKit test command
    wlk_parser = subparsers.add_parser("test-wlk", help="Test WhisperLiveKit streaming captions")
    wlk_parser.add_argument(
        "--server-url", "-s",
        default="ws://localhost:8000/asr",
        help="WhisperLiveKit server WebSocket URL",
    )
    wlk_parser.add_argument(
        "--model", "-m",
        default="base",
        choices=["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"],
        help="WLK model size (for auto-start server)",
    )
    wlk_parser.add_argument(
        "--model-path",
        help="Custom model path or Hugging Face repo ID (e.g., mlx-community/ivrit-ai-whisper-large-v3-turbo-mlx). Overrides --model.",
    )
    wlk_parser.add_argument(
        "--duration", "-d",
        type=int,
        default=60,
        help="Duration to capture captions (seconds)",
    )
    wlk_parser.add_argument(
        "--language", "-l",
        default="zh",
        help="Language code (zh, en, auto, etc.)",
    )
    wlk_parser.add_argument(
        "--output", "-o",
        help="Output file for captions",
    )
    wlk_parser.add_argument(
        "--auto-server",
        action="store_true",
        help="Automatically start WLK server",
    )
    wlk_parser.add_argument(
        "--diarization",
        action="store_true",
        help="Enable speaker identification (requires NeMo)",
    )

    # WLK + Camera integration test command
    wlk_cam_parser = subparsers.add_parser("test-wlk-camera", help="Test WLK + Virtual Camera integration with speaker overlay")
    wlk_cam_parser.add_argument(
        "--server-url", "-s",
        default="ws://localhost:8000/asr",
        help="WhisperLiveKit server WebSocket URL",
    )
    wlk_cam_parser.add_argument(
        "--duration", "-d",
        type=int,
        default=60,
        help="Duration to run the test (seconds)",
    )
    wlk_cam_parser.add_argument(
        "--language", "-l",
        default="zh",
        help="Language code (zh, en, auto, etc.)",
    )
    wlk_cam_parser.add_argument(
        "--diarization",
        action="store_true",
        help="Enable speaker identification (requires NeMo)",
    )
    wlk_cam_parser.add_argument(
        "--device",
        help="Virtual camera device (default: auto-detect)",
    )

    # WLK + TTS test command
    wlk_tts_parser = subparsers.add_parser("test-wlk-tts", help="Test WLK + TTS (text-to-speech) integration")
    wlk_tts_parser.add_argument(
        "--server-url", "-s",
        default="ws://localhost:8000/asr",
        help="WhisperLiveKit server WebSocket URL",
    )
    wlk_tts_parser.add_argument(
        "--duration", "-d",
        type=int,
        default=60,
        help="Duration to run the test (seconds)",
    )
    wlk_tts_parser.add_argument(
        "--language", "-l",
        default="zh",
        help="Language code (zh, en, auto, etc.)",
    )
    wlk_tts_parser.add_argument(
        "--no-tts",
        action="store_true",
        help="Disable TTS playback (captions only)",
    )
    wlk_tts_parser.add_argument(
        "--virtual-audio",
        action="store_true",
        help="Use virtual audio device for Zoom (requires setup)",
    )
    wlk_tts_parser.add_argument(
        "--audio-device",
        help="Virtual audio device name (e.g., 'BlackHole 2ch', 'Soundflower (2ch)')",
    )
    wlk_tts_parser.add_argument(
        "--input-device", "-i",
        type=int,
        help="Audio input device index (use 'list-audio-devices' to see available devices)",
    )
    wlk_tts_parser.add_argument(
        "--model",
        default="glm-tts",
        help="TTS model to use (default: glm-tts). Backend is auto-selected: "
             "glm-tts/glm-* -> GLM API (requires GLM_TTS_API_KEY), "
             "*mlx*/*qwen* -> MLX local TTS (experimental)",
    )
    wlk_tts_parser.add_argument(
        "--setup-audio",
        action="store_true",
        help="Show virtual audio setup instructions and exit",
    )

    # Enhanced WLK + Camera test command (with Chinese support)
    wlk_enhanced_parser = subparsers.add_parser("test-wlk-enhanced", help="Test Enhanced WLK + Virtual Camera with Chinese support and multiple styles")
    wlk_enhanced_parser.add_argument(
        "--server-url", "-s",
        default="ws://localhost:8000/asr",
        help="WhisperLiveKit server WebSocket URL",
    )
    wlk_enhanced_parser.add_argument(
        "--duration", "-d",
        type=int,
        default=60,
        help="Duration to run the test (seconds)",
    )
    wlk_enhanced_parser.add_argument(
        "--language", "-l",
        default="zh",
        help="Language code (zh, en, auto, etc.)",
    )
    wlk_enhanced_parser.add_argument(
        "--style",
        choices=["modern", "chat", "karaoke", "subtitle"],
        default="modern",
        help="Caption display style (default: modern)",
    )
    wlk_enhanced_parser.add_argument(
        "--diarization",
        action="store_true",
        default=True,
        help="Enable speaker identification (default: True)",
    )
    wlk_enhanced_parser.add_argument(
        "--no-diarization",
        action="store_true",
        help="Disable speaker identification",
    )

    # MLX Audio test command (STT & TTS)
    mlx_parser = subparsers.add_parser("test-mlx-audio", help="Test MLX Audio STT & TTS (Apple Silicon only)")
    mlx_parser.add_argument(
        "mode",
        choices=["stt", "tts", "full", "list-models"],
        help="Test mode: stt (speech-to-text), tts (text-to-speech), full (stt+tts), list-models",
    )
    mlx_parser.add_argument(
        "--audio", "-a",
        help="Input audio file path (for stt/full modes)",
    )
    mlx_parser.add_argument(
        "--mic", "-m",
        action="store_true",
        help="Use microphone for audio input (for stt/full modes)",
    )
    mlx_parser.add_argument(
        "--duration", "-d",
        type=float,
        default=5.0,
        help="Recording duration in seconds (for --mic, default: 5.0)",
    )
    mlx_parser.add_argument(
        "--text", "-t",
        help="Text to synthesize (for tts/full modes)",
    )
    mlx_parser.add_argument(
        "--stt-model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"],
        help="STT model size (default: base)",
    )
    mlx_parser.add_argument(
        "--tts-model",
        default="speecht5",
        help="TTS model name (default: speecht5)",
    )
    mlx_parser.add_argument(
        "--language", "-l",
        default="zh",
        help="Language code (default: zh)",
    )
    mlx_parser.add_argument(
        "--output", "-o",
        help="Output audio file path (for tts mode)",
    )
    mlx_parser.add_argument(
        "--play", "-p",
        action="store_true",
        help="Play synthesized audio (for tts mode)",
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


def cmd_list_audio_devices(args: argparse.Namespace):
    """Handle list-audio-devices command."""
    from zoom_ai.audio_captions import AudioCapturer
    AudioCapturer.list_devices()
    return 0


async def cmd_test_audio_captions(args: argparse.Namespace):
    """Handle test-audio-captions command."""
    from zoom_ai.audio_captions import AudioCaptionReader, AudioCaptionLogger

    logger.info("Testing audio caption reader (Whisper)...")
    logger.info(f"Model: {args.model}")
    logger.info(f"Duration: {args.duration} seconds")
    logger.info(f"Language: {args.language}")

    reader = AudioCaptionReader(
        model_size=args.model,
        language=args.language,
    )

    output_file = args.output or f"logs/audio_captions_{args.model}.txt"

    captions_logger = AudioCaptionLogger(output_file=output_file)
    reader.on_caption(captions_logger.on_caption)

    try:
        await reader.start()
        logger.info(f"Capturing audio captions for {args.duration} seconds...")
        logger.info("Speak into your microphone or play audio...")

        await asyncio.sleep(args.duration)

        captions = captions_logger.get_all_captions()
        logger.info(f"Captured {len(captions)} captions")
        logger.info(f"Captions saved to: {output_file}")

        return 0

    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Audio captions test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        await reader.stop()


async def cmd_test_wlk(args: argparse.Namespace):
    """Handle test-wlk command."""
    from zoom_ai.wlk_captions import (
        test_wlk_captions,
        test_wlk_with_server,
        WhisperLiveKitStreamer,
        WLKCaptionLogger,
    )

    logger.info("Testing WhisperLiveKit captions...")

    if args.auto_server:
        # Auto-start server
        model = args.model_path or args.model
        logger.info(f"Auto-starting WLK server with model: {model}")
        logger.info(f"Duration: {args.duration} seconds")
        logger.info(f"Language: {args.language}")
        logger.info(f"Diarization: {args.diarization}")

        return await test_wlk_with_server(
            model_size=args.model,
            model_path=args.model_path,
            language=args.language,
            duration=args.duration,
            diarization=args.diarization,
        )
    else:
        # Connect to existing server
        logger.info(f"Connecting to WLK server: {args.server_url}")
        logger.info(f"Duration: {args.duration} seconds")
        logger.info(f"Language: {args.language}")
        logger.info(f"Diarization: {args.diarization}")

        output_file = args.output or "logs/wlk_captions.txt"

        streamer = WhisperLiveKitStreamer(
            server_url=args.server_url,
            language=args.language,
            diarization=args.diarization,
        )

        caption_logger = WLKCaptionLogger(output_file=output_file)
        streamer.on_caption(caption_logger.on_caption)

        try:
            await streamer.start()
            logger.info(f"Capturing for {args.duration} seconds...")

            await asyncio.sleep(args.duration)

            captions = caption_logger.get_all_captions()
            logger.info(f"Captured {len(captions)} captions")
            logger.info(f"Captions saved to: {output_file}")

            return 0

        except KeyboardInterrupt:
            logger.info("Test interrupted by user")
            return 0
        except Exception as e:
            logger.error(f"WLK test failed: {e}")
            import traceback
            traceback.print_exc()
            return 1
        finally:
            await streamer.stop()


async def cmd_test_wlk_camera(args: argparse.Namespace):
    """Handle test-wlk-camera command - WLK + Virtual Camera integration."""
    from zoom_ai.wlk_camera_overlay import test_wlk_camera_overlay

    logger.info("Testing WLK + Virtual Camera integration...")
    logger.info(f"Server: {args.server_url}")
    logger.info(f"Duration: {args.duration} seconds")
    logger.info(f"Language: {args.language}")
    logger.info(f"Diarization: {args.diarization}")

    exit_code = await test_wlk_camera_overlay(
        wlk_server_url=args.server_url,
        language=args.language,
        diarization=args.diarization,
        duration=args.duration,
    )
    return exit_code


async def cmd_test_wlk_enhanced(args: argparse.Namespace):
    """Handle test-wlk-enhanced command - Enhanced WLK + Virtual Camera with Chinese support."""
    from zoom_ai.wlk_enhanced_overlay import test_enhanced_wlk_camera, CaptionStyle

    logger.info("Testing Enhanced WLK + Virtual Camera...")
    logger.info(f"Server: {args.server_url}")
    logger.info(f"Duration: {args.duration} seconds")
    logger.info(f"Language: {args.language}")
    logger.info(f"Style: {args.style}")
    logger.info(f"Diarization: {not args.no_diarization}")

    exit_code = await test_enhanced_wlk_camera(
        wlk_server_url=args.server_url,
        language=args.language,
        diarization=not args.no_diarization,
        duration=args.duration,
        style=args.style,
    )
    return exit_code


async def cmd_test_wlk_tts(args: argparse.Namespace):
    """Handle test-wlk-tts command - WLK + TTS integration."""
    from zoom_ai.wlk_tts_overlay import test_wlk_tts
    import platform

    # Show setup instructions if requested
    if hasattr(args, 'setup_audio') and args.setup_audio:
        from zoom_ai.virtual_audio import setup_virtual_audio_macos, setup_virtual_audio_linux
        print("\n" + "="*60)
        print("虚拟音频设置指南 (Virtual Audio Setup)")
        print("="*60)
        if platform.system() == "Darwin":
            print(setup_virtual_audio_macos())
        elif platform.system() == "Linux":
            print(setup_virtual_audio_linux())
        else:
            print(f"暂不支持 {platform.system()} 平台的虚拟音频设置")
        print("\n设置完成后，重新运行命令并添加 --virtual-audio 参数")
        return 0

    logger.info("Testing WLK + TTS Integration...")
    logger.info(f"Server: {args.server_url}")
    logger.info(f"Duration: {args.duration} seconds")
    logger.info(f"Language: {args.language}")
    logger.info(f"TTS: {'Disabled' if args.no_tts else 'Enabled'}")

    # Check if virtual audio is enabled
    use_virtual_audio = hasattr(args, 'virtual_audio') and args.virtual_audio
    if use_virtual_audio:
        logger.info(f"Virtual Audio: Enabled (device: {args.audio_device or 'auto'})")

    # Get input device
    input_device = getattr(args, 'input_device', None)
    if input_device is not None:
        logger.info(f"Input device: {input_device}")

    # Get TTS model
    tts_model = getattr(args, 'model', 'glm-tts')
    logger.info(f"TTS Model: {tts_model}")

    exit_code = await test_wlk_tts(
        wlk_server_url=args.server_url,
        language=args.language,
        diarization=True,
        duration=args.duration,
        auto_tts=not args.no_tts,
        use_virtual_audio=use_virtual_audio,
        virtual_audio_device=getattr(args, 'audio_device', None),
        input_device=input_device,
        model=tts_model,
    )
    return exit_code


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


def cmd_test_mlx_audio(args: argparse.Namespace) -> int:
    """Handle test-mlx-audio command - MLX Audio STT & TTS testing."""
    from zoom_ai.test_mlx_audio import (
        check_installation,
        MLXSTTTester,
        MLXTTSTester,
        MLXFullPipeline,
        list_models,
    )

    logger.info("Testing MLX Audio...")

    # list-models 模式不需要检查安装
    if args.mode == "list-models":
        print("可用的 STT 模型:")
        for name, path in list_models().get("stt", {}).items():
            print(f"  {name}: {path}")
        print("\n可用的 TTS 模型:")
        for name, path in list_models().get("tts", {}).items():
            print(f"  {name}: {path}")
        print("\n注意: MLX Audio 仅支持 Apple Silicon (M系列芯片)")
        return 0

    # 其他模式需要检查安装
    if not check_installation():
        logger.error("请先安装 MLX Audio 依赖:")
        logger.error("  uv add --optional mlx mlx-audio sounddevice scipy")
        logger.error("  uv sync --extra mlx")
        return 1

    try:
        if args.mode == "stt":
            stt = MLXSTTTester(model=args.stt_model, language=args.language)
            if args.audio:
                text = stt.transcribe_file(args.audio)
                return 0 if text else 1
            elif args.mic:
                text = stt.transcribe_mic(duration=args.duration)
                return 0 if text else 1
            else:
                logger.error("请指定 --audio 或 --mic")
                return 1

        elif args.mode == "tts":
            tts = MLXTTSTester(model=args.tts_model)
            text = args.text or "你好，这是测试。"
            if args.play:
                result = tts.synthesize_and_play(text)
            else:
                output = tts.synthesize(text, args.output)
                result = output is not None
            return 0 if result else 1

        elif args.mode == "full":
            pipeline = MLXFullPipeline(
                stt_model=args.stt_model,
                tts_model=args.tts_model,
                language=args.language,
            )
            if args.audio:
                result = pipeline.test_with_file(args.audio)
            elif args.mic:
                result = pipeline.test_with_mic(duration=args.duration)
            else:
                logger.error("请指定 --audio 或 --mic")
                return 1
            return 0 if result else 1

    except ImportError as e:
        logger.error(f"MLX Audio 导入失败: {e}")
        logger.info("请运行: uv add --optional mlx mlx-audio sounddevice scipy && uv sync --extra mlx")
        return 1
    except Exception as e:
        logger.error(f"MLX Audio 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


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
    elif args.command == "list-audio-devices":
        exit_code = cmd_list_audio_devices(args)
        sys.exit(exit_code)
    elif args.command == "test-audio-captions":
        exit_code = asyncio.run(cmd_test_audio_captions(args))
        sys.exit(exit_code)
    elif args.command == "test-wlk":
        exit_code = asyncio.run(cmd_test_wlk(args))
        sys.exit(exit_code)
    elif args.command == "test-wlk-camera":
        exit_code = asyncio.run(cmd_test_wlk_camera(args))
        sys.exit(exit_code)
    elif args.command == "test-wlk-enhanced":
        exit_code = asyncio.run(cmd_test_wlk_enhanced(args))
        sys.exit(exit_code)
    elif args.command == "test-wlk-tts":
        exit_code = asyncio.run(cmd_test_wlk_tts(args))
        sys.exit(exit_code)
    elif args.command == "test-mlx-audio":
        exit_code = cmd_test_mlx_audio(args)
        sys.exit(exit_code)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
