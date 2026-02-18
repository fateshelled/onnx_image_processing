"""
Camera wrapper classes for different camera backends.

Provides a unified interface for OpenCV cameras and RealSense cameras.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import cv2
import numpy as np


class BaseCamera(ABC):
    """
    Abstract base class for camera interfaces.

    Provides a unified interface for different camera backends
    (OpenCV, RealSense, etc.).
    """

    @abstractmethod
    def open(self) -> bool:
        """
        Open the camera.

        Returns:
            True if camera opened successfully, False otherwise
        """
        pass

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the camera.

        Returns:
            Tuple of (success, frame)
            - success: True if frame was read successfully
            - frame: BGR image array (H, W, 3) or None if failed
        """
        pass

    @abstractmethod
    def release(self) -> None:
        """Release camera resources."""
        pass

    @abstractmethod
    def is_opened(self) -> bool:
        """
        Check if camera is opened.

        Returns:
            True if camera is opened, False otherwise
        """
        pass

    @abstractmethod
    def get_fps(self) -> float:
        """
        Get camera FPS.

        Returns:
            Camera FPS (frames per second)
        """
        pass

    @abstractmethod
    def get_resolution(self) -> Tuple[int, int]:
        """
        Get camera resolution.

        Returns:
            Tuple of (width, height)
        """
        pass


class OpenCVCamera(BaseCamera):
    """
    OpenCV camera wrapper.

    Wraps cv2.VideoCapture for standard USB/webcam cameras.
    """

    def __init__(self, device_id: int = 0):
        """
        Initialize OpenCV camera.

        Args:
            device_id: Camera device ID (default: 0)
        """
        self.device_id = device_id
        self.cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        """Open the camera."""
        self.cap = cv2.VideoCapture(self.device_id)
        return self.cap.isOpened()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the camera."""
        if self.cap is None or not self.cap.isOpened():
            return False, None
        return self.cap.read()

    def release(self) -> None:
        """Release camera resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def is_opened(self) -> bool:
        """Check if camera is opened."""
        return self.cap is not None and self.cap.isOpened()

    def get_fps(self) -> float:
        """Get camera FPS."""
        if self.cap is None:
            return 30.0
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        return fps if fps > 0 else 30.0

    def get_resolution(self) -> Tuple[int, int]:
        """Get camera resolution."""
        if self.cap is None:
            return (640, 480)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)

    def set_resolution(self, width: int, height: int) -> bool:
        """
        Set camera resolution.

        Args:
            width: Target width
            height: Target height

        Returns:
            True if resolution was set successfully
        """
        if self.cap is None:
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        return True

    def __repr__(self) -> str:
        return f"OpenCVCamera(device_id={self.device_id})"


class RealSenseCamera(BaseCamera):
    """
    Intel RealSense camera wrapper.

    Wraps pyrealsense2 for RealSense D400/D500 series cameras.
    Provides RGB and optionally depth streams.
    """

    def __init__(
        self,
        device_id: Optional[str] = None,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        enable_depth: bool = False,
    ):
        """
        Initialize RealSense camera.

        Args:
            device_id: RealSense device serial number (None for first available)
            width: RGB stream width (default: 640)
            height: RGB stream height (default: 480)
            fps: Stream framerate (default: 30)
            enable_depth: Enable depth stream (default: False)
        """
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self.enable_depth = enable_depth

        self.pipeline = None
        self.config = None
        self.align = None

        # Check if pyrealsense2 is available
        try:
            import pyrealsense2 as rs
            self.rs = rs
        except ImportError:
            raise ImportError(
                "pyrealsense2 is not installed. "
                "Install it with: pip install pyrealsense2"
            )

    def open(self) -> bool:
        """Open the camera."""
        try:
            self.pipeline = self.rs.pipeline()
            self.config = self.rs.config()

            # Enable device by serial number if specified
            if self.device_id is not None:
                self.config.enable_device(self.device_id)

            # Enable RGB stream
            self.config.enable_stream(
                self.rs.stream.color,
                self.width,
                self.height,
                self.rs.format.bgr8,
                self.fps,
            )

            # Enable depth stream if requested
            if self.enable_depth:
                self.config.enable_stream(
                    self.rs.stream.depth,
                    self.width,
                    self.height,
                    self.rs.format.z16,
                    self.fps,
                )
                # Create alignment object to align depth to color
                self.align = self.rs.align(self.rs.stream.color)

            # Start pipeline
            self.pipeline.start(self.config)
            return True

        except Exception as e:
            print(f"Failed to open RealSense camera: {e}")
            return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the camera.

        Returns RGB image. Use read_rgbd() for RGB+Depth.
        """
        if self.pipeline is None:
            return False, None

        try:
            # Wait for frames
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)

            # Get color frame
            color_frame = frames.get_color_frame()
            if not color_frame:
                return False, None

            # Convert to numpy array
            color_image = np.asanyarray(color_frame.get_data())
            return True, color_image

        except Exception as e:
            print(f"Failed to read frame: {e}")
            return False, None

    def read_rgbd(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Read RGB and depth frames from the camera.

        Returns:
            Tuple of (success, rgb_image, depth_image)
            - success: True if frames were read successfully
            - rgb_image: BGR image array (H, W, 3) or None
            - depth_image: Depth image array (H, W) in millimeters or None
        """
        if self.pipeline is None:
            return False, None, None

        if not self.enable_depth:
            ret, rgb = self.read()
            return ret, rgb, None

        try:
            # Wait for frames
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)

            # Align depth to color
            if self.align is not None:
                frames = self.align.process(frames)

            # Get frames
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame:
                return False, None, None

            # Convert to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data()) if depth_frame else None

            return True, color_image, depth_image

        except Exception as e:
            print(f"Failed to read RGBD frames: {e}")
            return False, None, None

    def get_intrinsics(self):
        """
        Get camera intrinsic parameters.

        Returns:
            pyrealsense2.intrinsics object or None if not available
        """
        if self.pipeline is None:
            return None

        try:
            # Get active profile
            profile = self.pipeline.get_active_profile()
            color_stream = profile.get_stream(self.rs.stream.color)
            intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
            return intrinsics
        except Exception:
            return None

    def get_camera_intrinsics(self):
        """
        Get camera intrinsic parameters as CameraIntrinsics object.

        Returns:
            CameraIntrinsics object or None if not available
        """
        intrinsics = self.get_intrinsics()
        if intrinsics is None:
            return None

        # Import here to avoid circular dependency
        from .pose_estimation import CameraIntrinsics

        return CameraIntrinsics(
            fx=intrinsics.fx,
            fy=intrinsics.fy,
            cx=intrinsics.ppx,
            cy=intrinsics.ppy,
            width=intrinsics.width,
            height=intrinsics.height,
        )

    def release(self) -> None:
        """Release camera resources."""
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None
        self.config = None
        self.align = None

    def is_opened(self) -> bool:
        """Check if camera is opened."""
        return self.pipeline is not None

    def get_fps(self) -> float:
        """Get camera FPS."""
        return float(self.fps)

    def get_resolution(self) -> Tuple[int, int]:
        """Get camera resolution."""
        return (self.width, self.height)

    def __repr__(self) -> str:
        return (
            f"RealSenseCamera(device_id={self.device_id}, "
            f"resolution={self.width}x{self.height}, "
            f"fps={self.fps}, depth={self.enable_depth})"
        )


def create_camera(
    backend: str = "opencv",
    device_id: int = 0,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    enable_depth: bool = False,
) -> BaseCamera:
    """
    Create a camera instance based on backend type.

    Args:
        backend: Camera backend ("opencv" or "realsense")
        device_id: Camera device ID (for OpenCV) or serial number (for RealSense)
        width: Camera resolution width
        height: Camera resolution height
        fps: Camera framerate
        enable_depth: Enable depth stream (RealSense only)

    Returns:
        Camera instance

    Raises:
        ValueError: If backend is not supported
    """
    backend = backend.lower()

    if backend == "opencv":
        camera = OpenCVCamera(device_id=device_id)
        if not camera.open():
            raise RuntimeError(f"Failed to open OpenCV camera {device_id}")
        # Try to set resolution
        camera.set_resolution(width, height)
        return camera

    elif backend == "realsense":
        device_serial = str(device_id) if device_id != 0 else None
        camera = RealSenseCamera(
            device_id=device_serial,
            width=width,
            height=height,
            fps=fps,
            enable_depth=enable_depth,
        )
        if not camera.open():
            raise RuntimeError("Failed to open RealSense camera")
        return camera

    else:
        raise ValueError(
            f"Unsupported camera backend: {backend}. "
            f"Supported backends: opencv, realsense"
        )
