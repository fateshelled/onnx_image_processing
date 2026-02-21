"""
Camera wrapper classes for different camera backends.

Provides a unified interface for OpenCV cameras and RealSense cameras.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import cv2
import numpy as np

# Set up module logger
logger = logging.getLogger(__name__)


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
        if self.cap is None or not self.cap.isOpened():
            return 0.0
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        return fps if fps > 0 else 0.0

    def get_resolution(self) -> Tuple[int, int]:
        """Get camera resolution."""
        if self.cap is None or not self.cap.isOpened():
            return (0, 0)
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
            logger.error(f"Failed to open RealSense camera: {e}")
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
            logger.error(f"Failed to read frame: {e}")
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
            logger.error(f"Failed to read RGBD frames: {e}")
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


class OrbbecCamera(BaseCamera):
    """
    Orbbec camera wrapper.

    Wraps pyorbbecsdk for Orbbec Astra/Femto/Gemini series cameras.
    Provides RGB and optionally depth streams.
    """

    def __init__(
        self,
        device_id: Optional[int] = None,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        enable_depth: bool = False,
    ):
        """
        Initialize Orbbec camera.

        Args:
            device_id: Camera device index (None for first available)
            width: RGB stream width (default: 640)
            height: RGB stream height (default: 480)
            fps: Stream framerate (default: 30)
            enable_depth: Enable depth stream (default: False)
        """
        self.device_id = device_id if device_id is not None else 0
        self.width = width
        self.height = height
        self.fps = fps
        self.enable_depth = enable_depth

        self.pipeline = None
        self.config = None
        self.align = None

        # Check if pyorbbecsdk is available
        try:
            from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, OBAlignMode
            self.Pipeline = Pipeline
            self.Config = Config
            self.OBSensorType = OBSensorType
            self.OBFormat = OBFormat
            self.OBAlignMode = OBAlignMode
        except ImportError:
            raise ImportError(
                "pyorbbecsdk is not installed. "
                "Install it with: pip install pyorbbecsdk"
            )

    def open(self) -> bool:
        """Open the camera."""
        try:
            # Create pipeline and config
            self.pipeline = self.Pipeline()
            self.config = self.Config()

            # Get device
            device_list = self.pipeline.get_device_list()
            if device_list.get_count() == 0:
                raise RuntimeError("No Orbbec devices found")

            if self.device_id >= device_list.get_count():
                raise RuntimeError(f"Device index {self.device_id} out of range (found {device_list.get_count()} devices)")

            # Enable color stream
            color_profiles = self.pipeline.get_stream_profile_list(self.OBSensorType.COLOR_SENSOR)
            if color_profiles is None:
                raise RuntimeError("No color profiles found")

            # Find matching profile
            color_profile = None
            for i in range(color_profiles.get_count()):
                profile = color_profiles.get_profile(i)
                if (profile.get_width() == self.width and
                    profile.get_height() == self.height and
                    profile.get_fps() == self.fps):
                    color_profile = profile
                    break

            if color_profile is None:
                # Try to find any profile and use it
                if color_profiles.get_count() > 0:
                    color_profile = color_profiles.get_profile(0)
                    requested_profile = f"{self.width}x{self.height}@{self.fps}fps"
                    self.width = color_profile.get_width()
                    self.height = color_profile.get_height()
                    self.fps = color_profile.get_fps()
                    logger.warning(f"Requested profile {requested_profile} not found. "
                                 f"Using available profile: {self.width}x{self.height}@{self.fps}fps")
                else:
                    raise RuntimeError("No suitable color profile found")

            self.config.enable_stream(color_profile)

            # Enable depth stream if requested
            if self.enable_depth:
                depth_profiles = self.pipeline.get_stream_profile_list(self.OBSensorType.DEPTH_SENSOR)
                if depth_profiles and depth_profiles.get_count() > 0:
                    # Find matching depth profile
                    depth_profile = None
                    for i in range(depth_profiles.get_count()):
                        profile = depth_profiles.get_profile(i)
                        if (profile.get_width() == self.width and
                            profile.get_height() == self.height and
                            profile.get_fps() == self.fps):
                            depth_profile = profile
                            break

                    if depth_profile is None:
                        depth_profile = depth_profiles.get_profile(0)

                    self.config.enable_stream(depth_profile)
                    # Enable hardware alignment
                    self.config.set_align_mode(self.OBAlignMode.HW_MODE)

            # Start pipeline
            self.pipeline.start(self.config)
            return True

        except Exception as e:
            logger.error(f"Failed to open Orbbec camera: {e}")
            return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the camera.

        Returns RGB image. Use read_rgbd() for RGB+Depth.
        """
        if self.pipeline is None:
            return False, None

        try:
            # Wait for frames (timeout 1000ms)
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            if frames is None:
                return False, None

            # Get color frame
            color_frame = frames.get_color_frame()
            if color_frame is None:
                return False, None

            # Convert to numpy array
            color_image = np.asanyarray(color_frame.get_data())

            # Convert RGB to BGR for OpenCV compatibility
            if len(color_image.shape) == 3 and color_image.shape[2] == 3:
                color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

            return True, color_image

        except Exception as e:
            logger.error(f"Failed to read frame: {e}")
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
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            if frames is None:
                return False, None, None

            # Get frames
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if color_frame is None:
                return False, None, None

            # Convert to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            if len(color_image.shape) == 3 and color_image.shape[2] == 3:
                color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

            depth_image = np.asanyarray(depth_frame.get_data()) if depth_frame else None

            return True, color_image, depth_image

        except Exception as e:
            logger.error(f"Failed to read RGBD frames: {e}")
            return False, None, None

    def get_intrinsics(self):
        """
        Get camera intrinsic parameters.

        Returns:
            Camera intrinsics object or None if not available
        """
        if self.pipeline is None:
            return None

        try:
            # Get camera parameters from color stream
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            if frames is None:
                return None

            color_frame = frames.get_color_frame()
            if color_frame is None:
                return None

            camera_params = color_frame.get_camera_param()
            return camera_params
        except Exception:
            return None

    def get_camera_intrinsics(self):
        """
        Get camera intrinsic parameters as CameraIntrinsics object.

        Returns:
            CameraIntrinsics object or None if not available
        """
        camera_params = self.get_intrinsics()
        if camera_params is None:
            return None

        # Import here to avoid circular dependency
        from .pose_estimation import CameraIntrinsics

        # Orbbec provides RGB camera intrinsics
        rgb_intrinsic = camera_params.rgb_intrinsic

        return CameraIntrinsics(
            fx=rgb_intrinsic.fx,
            fy=rgb_intrinsic.fy,
            cx=rgb_intrinsic.cx,
            cy=rgb_intrinsic.cy,
            width=rgb_intrinsic.width,
            height=rgb_intrinsic.height,
        )

    def release(self) -> None:
        """Release camera resources."""
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None
        self.config = None

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
            f"OrbbecCamera(device_id={self.device_id}, "
            f"resolution={self.width}x{self.height}, "
            f"fps={self.fps}, depth={self.enable_depth})"
        )


class OAKCamera(BaseCamera):
    """
    Luxonis OAK-D camera wrapper.

    Wraps depthai for OAK-D series cameras (OAK-D, OAK-D Lite, OAK-D Pro).
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
        Initialize OAK-D camera.

        Args:
            device_id: Device MxID (None for first available)
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
        self.device = None
        self.q_rgb = None
        self.q_depth = None

        # Check if depthai is available
        try:
            import depthai as dai
            self.dai = dai
        except ImportError:
            raise ImportError(
                "depthai is not installed. "
                "Install it with: pip install depthai"
            )

    def open(self) -> bool:
        """Open the camera."""
        try:
            # Create pipeline
            self.pipeline = self.dai.Pipeline()

            # Create color camera node
            cam_rgb = self.pipeline.create(self.dai.node.ColorCamera)
            cam_rgb.setBoardSocket(self.dai.CameraBoardSocket.CAM_A)
            cam_rgb.setResolution(self.dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            cam_rgb.setVideoSize(self.width, self.height)
            cam_rgb.setFps(self.fps)
            cam_rgb.setInterleaved(False)
            cam_rgb.setColorOrder(self.dai.ColorCameraProperties.ColorOrder.BGR)

            # Create output
            xout_rgb = self.pipeline.create(self.dai.node.XLinkOut)
            xout_rgb.setStreamName("rgb")
            cam_rgb.video.link(xout_rgb.input)

            # Create depth if requested
            if self.enable_depth:
                # Create stereo depth node
                mono_left = self.pipeline.create(self.dai.node.MonoCamera)
                mono_right = self.pipeline.create(self.dai.node.MonoCamera)
                stereo = self.pipeline.create(self.dai.node.StereoDepth)

                mono_left.setResolution(self.dai.MonoCameraProperties.SensorResolution.THE_400_P)
                mono_left.setBoardSocket(self.dai.CameraBoardSocket.CAM_B)
                mono_right.setResolution(self.dai.MonoCameraProperties.SensorResolution.THE_400_P)
                mono_right.setBoardSocket(self.dai.CameraBoardSocket.CAM_C)

                stereo.setDefaultProfilePreset(self.dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
                stereo.setLeftRightCheck(True)
                stereo.setDepthAlign(self.dai.CameraBoardSocket.CAM_A)

                mono_left.out.link(stereo.left)
                mono_right.out.link(stereo.right)

                xout_depth = self.pipeline.create(self.dai.node.XLinkOut)
                xout_depth.setStreamName("depth")
                stereo.depth.link(xout_depth.input)

            # Connect to device
            if self.device_id is not None:
                device_info = self.dai.DeviceInfo(self.device_id)
                self.device = self.dai.Device(self.pipeline, device_info)
            else:
                self.device = self.dai.Device(self.pipeline)

            # Get output queues
            self.q_rgb = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            if self.enable_depth:
                self.q_depth = self.device.getOutputQueue(name="depth", maxSize=4, blocking=False)

            return True

        except Exception as e:
            logger.error(f"Failed to open OAK-D camera: {e}")
            return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the camera.

        Returns RGB image. Use read_rgbd() for RGB+Depth.
        """
        if self.q_rgb is None:
            return False, None

        try:
            in_rgb = self.q_rgb.tryGet()
            if in_rgb is None:
                return False, None

            # Get BGR frame
            frame = in_rgb.getCvFrame()
            return True, frame

        except Exception as e:
            logger.error(f"Failed to read frame: {e}")
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
        if self.q_rgb is None:
            return False, None, None

        if not self.enable_depth:
            ret, rgb = self.read()
            return ret, rgb, None

        try:
            in_rgb = self.q_rgb.tryGet()
            in_depth = self.q_depth.tryGet() if self.q_depth else None

            if in_rgb is None:
                return False, None, None

            # Get frames
            rgb_frame = in_rgb.getCvFrame()
            depth_frame = in_depth.getFrame() if in_depth else None

            return True, rgb_frame, depth_frame

        except Exception as e:
            logger.error(f"Failed to read RGBD frames: {e}")
            return False, None, None

    def get_intrinsics(self):
        """
        Get camera intrinsic parameters.

        Returns:
            Camera calibration data or None if not available
        """
        if self.device is None:
            return None

        try:
            calib_data = self.device.readCalibration()
            return calib_data
        except Exception:
            return None

    def get_camera_intrinsics(self):
        """
        Get camera intrinsic parameters as CameraIntrinsics object.

        Returns:
            CameraIntrinsics object or None if not available
        """
        calib_data = self.get_intrinsics()
        if calib_data is None:
            return None

        # Import here to avoid circular dependency
        from .pose_estimation import CameraIntrinsics

        try:
            # Get RGB camera intrinsics
            intrinsics = calib_data.getCameraIntrinsics(
                self.dai.CameraBoardSocket.CAM_A,
                self.width,
                self.height
            )

            # intrinsics is a list: [fx, fy, cx, cy]
            # or a 3x3 matrix
            if len(intrinsics) == 2:  # It's a tuple of (intrinsic_matrix, distortion)
                K = intrinsics[0]  # 3x3 intrinsic matrix
                fx = K[0][0]
                fy = K[1][1]
                cx = K[0][2]
                cy = K[1][2]
            elif hasattr(intrinsics, '__getitem__') and len(intrinsics) == 3:
                # 3x3 intrinsic matrix
                fx = intrinsics[0][0]
                fy = intrinsics[1][1]
                cx = intrinsics[0][2]
                cy = intrinsics[1][2]
            else:
                raise ValueError(
                    f"Invalid camera intrinsics format. Expected either:\n"
                    f"  - Tuple of (3x3 intrinsic_matrix, distortion)\n"
                    f"  - 3x3 intrinsic matrix\n"
                    f"Got: {type(intrinsics)}"
                )

            return CameraIntrinsics(
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                width=self.width,
                height=self.height,
            )
        except Exception as e:
            logger.error(f"Failed to get camera intrinsics: {e}")
            return None

    def release(self) -> None:
        """Release camera resources."""
        if self.device is not None:
            self.device.close()
            self.device = None
        self.pipeline = None
        self.q_rgb = None
        self.q_depth = None

    def is_opened(self) -> bool:
        """Check if camera is opened."""
        return self.device is not None

    def get_fps(self) -> float:
        """Get camera FPS."""
        return float(self.fps)

    def get_resolution(self) -> Tuple[int, int]:
        """Get camera resolution."""
        return (self.width, self.height)

    def __repr__(self) -> str:
        return (
            f"OAKCamera(device_id={self.device_id}, "
            f"resolution={self.width}x{self.height}, "
            f"fps={self.fps}, depth={self.enable_depth})"
        )


def create_camera(
    backend: str = "opencv",
    device_id: Union[int, str] = 0,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    enable_depth: bool = False,
) -> BaseCamera:
    """
    Create a camera instance based on backend type.

    Args:
        backend: Camera backend ("opencv", "realsense", "orbbec", or "oak")
        device_id: Camera device ID (int for OpenCV, int/str for others)
                   - OpenCV/Orbbec: integer device index
                   - RealSense: serial number (string) or 0 for default
                   - OAK-D: MxID (string) or 0 for default
        width: Camera resolution width
        height: Camera resolution height
        fps: Camera framerate
        enable_depth: Enable depth stream (RealSense/Orbbec/OAK only)

    Returns:
        Camera instance

    Raises:
        ValueError: If backend is not supported
    """
    backend = backend.lower()

    if backend == "opencv":
        # OpenCV expects integer device ID
        device_int = int(device_id) if isinstance(device_id, str) else device_id
        camera = OpenCVCamera(device_id=device_int)
        if not camera.open():
            raise RuntimeError(f"Failed to open OpenCV camera {device_id}")
        # Try to set resolution
        camera.set_resolution(width, height)
        return camera

    elif backend == "realsense":
        device_serial = str(device_id) if str(device_id) != '0' else None
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

    elif backend == "orbbec":
        camera = OrbbecCamera(
            device_id=device_id,
            width=width,
            height=height,
            fps=fps,
            enable_depth=enable_depth,
        )
        if not camera.open():
            raise RuntimeError("Failed to open Orbbec camera")
        return camera

    elif backend == "oak" or backend == "oak-d":
        device_mxid = str(device_id) if str(device_id) != '0' else None
        camera = OAKCamera(
            device_id=device_mxid,
            width=width,
            height=height,
            fps=fps,
            enable_depth=enable_depth,
        )
        if not camera.open():
            raise RuntimeError("Failed to open OAK-D camera")
        return camera

    else:
        raise ValueError(
            f"Unsupported camera backend: {backend}. "
            f"Supported backends: opencv, realsense, orbbec, oak"
        )
