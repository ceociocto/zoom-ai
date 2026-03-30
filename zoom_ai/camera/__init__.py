"""
Virtual camera management using v4l2loopback (Linux) or pyvirtualcam (macOS).

This module provides functionality to stream video frames to a virtual
camera device that can be used by Zoom or other video conferencing apps.
"""

import asyncio
import platform
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from loguru import logger

from zoom_ai.config import settings


class VirtualCamera:
    """
    Manages a virtual camera device for streaming video frames.

    Uses v4l2loopback on Linux or pyvirtualcam on macOS.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
    ):
        """
        Initialize the virtual camera.

        Args:
            device: Device path (e.g., /dev/video0). Only used on Linux.
            width: Frame width in pixels.
            height: Frame height in pixels.
            fps: Target frames per second.
        """
        self.system = platform.system()
        self.device = device or settings.virtual_camera_device_auto
        self.width = width
        self.height = height
        self.fps = fps
        self._writer: Optional[cv2.VideoWriter] = None
        self._pyvirtualcam_camera = None
        self._is_running = False
        self._frame_count = 0

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def open(self) -> bool:
        """
        Open the virtual camera device for writing.

        Returns:
            True if successful, False otherwise.
        """
        if self.system == "Linux":
            return self._open_linux()
        elif self.system == "Darwin":
            return self._open_macos()
        else:
            logger.error(f"Unsupported platform: {self.system}")
            return False

    def _open_linux(self) -> bool:
        """Open virtual camera on Linux using v4l2loopback."""
        try:
            # Check if device exists
            if not Path(self.device).exists():
                logger.error(f"Virtual camera device not found: {self.device}")
                logger.info("Create virtual camera with: sudo modprobe v4l2loopback devices=4 exclusive_caps=1")
                return False

            # Open device using VideoWriter with V4L2 backend
            fourcc = cv2.VideoWriter_fourcc(*'YUYV')  # YUYV format for v4l2
            self._writer = cv2.VideoWriter(
                self.device,
                cv2.CAP_V4L2,  # V4L2 backend
                fourcc,
                self.fps,
                (self.width, self.height),
                isColor=True
            )

            if not self._writer.isOpened():
                logger.error(f"Failed to open virtual camera: {self.device}")
                return False

            self._is_running = True
            logger.info(f"Virtual camera opened (Linux): {self.device} @ {self.width}x{self.height}@{self.fps}fps")
            return True

        except Exception as e:
            logger.error(f"Error opening virtual camera (Linux): {e}")
            return False

    def _open_macos(self) -> bool:
        """Open virtual camera on macOS using pyvirtualcam."""
        try:
            import pyvirtualcam

            # Try to use OBS Virtual Camera or other available camera
            self._pyvirtualcam_camera = pyvirtualcam.Camera(
                width=self.width,
                height=self.height,
                fps=self.fps,
                fmt=pyvirtualcam.PixelFormat.BGR,
            )

            self._is_running = True
            logger.info(f"Virtual camera opened (macOS): {self._pyvirtualcam_camera.device} @ {self.width}x{self.height}@{self.fps}fps")
            return True

        except ImportError:
            logger.error("pyvirtualcam not installed. Install with: pip install pyvirtualcam")
            logger.info("Also make sure OBS Virtual Camera is installed and enabled in OBS")
            return False
        except Exception as e:
            logger.error(f"Error opening virtual camera (macOS): {e}")
            logger.info("Make sure OBS Virtual Camera is started in OBS Studio")
            return False

    def close(self):
        """Close the virtual camera device."""
        if self.system == "Linux" and self._writer is not None:
            self._writer.release()
            self._writer = None
        elif self.system == "Darwin" and self._pyvirtualcam_camera is not None:
            self._pyvirtualcam_camera.close()
            self._pyvirtualcam_camera = None

        self._is_running = False
        logger.info(f"Virtual camera closed ({self.system})")

    def write(self, frame: np.ndarray) -> bool:
        """
        Write a single frame to the virtual camera.

        Args:
            frame: Frame as numpy array (BGR format, HxWx3).

        Returns:
            True if successful, False otherwise.
        """
        if not self._is_running:
            return False

        try:
            # Resize frame if needed
            if frame.shape[:2] != (self.height, self.width):
                frame = cv2.resize(frame, (self.width, self.height))

            # Ensure BGR format
            if len(frame.shape) == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            elif len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            if self.system == "Linux" and self._writer is not None:
                self._writer.write(frame)
                self._frame_count += 1
                return True
            elif self.system == "Darwin" and self._pyvirtualcam_camera is not None:
                self._pyvirtualcam_camera.send(frame)
                self._pyvirtualcam_camera.sleep_until_next_frame()
                self._frame_count += 1
                return True

            return False

        except Exception as e:
            logger.error(f"Error writing frame: {e}")
            return False

    async def write_async(self, frame: np.ndarray) -> bool:
        """
        Async wrapper for writing frames.

        Args:
            frame: Frame as numpy array.

        Returns:
            True if successful, False otherwise.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.write, frame)

    @staticmethod
    def list_devices() -> list[str]:
        """
        List available virtual camera devices.

        Returns:
            List of device paths or names.
        """
        system = platform.system()

        if system == "Linux":
            devices = []
            for i in range(10):  # Check /dev/video0-9
                device = f"/dev/video{i}"
                if Path(device).exists():
                    devices.append(device)
            return devices
        elif system == "Darwin":
            try:
                import pyvirtualcam
                # pyvirtualcam doesn't have a direct list method, return common names
                return ["OBS Virtual Camera", "Virtual Camera"]
            except ImportError:
                return []
        else:
            return []

    @staticmethod
    def create_v4l2loopback_devices(count: int = 4) -> bool:
        """
        Create v4l2loopback devices (Linux only).

        Note: Requires sudo privileges.

        Args:
            count: Number of devices to create.

        Returns:
            True if successful, False otherwise.
        """
        if platform.system() != "Linux":
            logger.warning("v4l2loopback is only available on Linux")
            return False

        import subprocess

        try:
            # Load module with specified device count
            result = subprocess.run(
                ["sudo", "modprobe", "v4l2loopback",
                 f"devices={count}", "exclusive_caps=1"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                logger.info(f"Created {count} v4l2loopback devices")
                return True
            else:
                logger.error(f"Failed to create devices: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Error creating v4l2loopback devices: {e}")
            return False


class FrameBuffer:
    """
    Thread-safe frame buffer for streaming to virtual camera.
    """

    def __init__(self, max_size: int = 2):
        """
        Initialize frame buffer.

        Args:
            max_size: Maximum number of frames to buffer.
        """
        self._buffer: list[np.ndarray] = []
        self._max_size = max_size
        self._latest_frame: Optional[np.ndarray] = None

    def put(self, frame: np.ndarray):
        """Put a frame in the buffer."""
        self._latest_frame = frame.copy()
        if len(self._buffer) >= self._max_size:
            self._buffer.pop(0)
        self._buffer.append(frame)

    def get(self) -> Optional[np.ndarray]:
        """Get the latest frame from the buffer."""
        return self._latest_frame

    def clear(self):
        """Clear the buffer."""
        self._buffer.clear()
        self._latest_frame = None
