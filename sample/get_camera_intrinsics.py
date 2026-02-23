#!/usr/bin/env python3
"""
Camera intrinsic parameter query tool.

Outputs fx, fy, cx, cy for use with visual_odometry.py and other tools.

Supported camera types
  realsense  : Intel RealSense (D400 / L500 series) via pyrealsense2.
               Reads calibrated intrinsics directly from the device firmware.
  oakd       : Luxonis OAK-D series via depthai.
               Reads calibrated intrinsics stored on the device.
  orbbec     : Orbbec (Femto Bolt / Femto Mega / Astra 2 etc.) via pyorbbecsdk.
               Reads calibrated intrinsics by starting a brief pipeline.
  opencv     : Generic USB/built-in webcam via OpenCV.
               Computes approximate intrinsics from resolution + assumed FOV.

Usage examples
  # --- RealSense ---
  python sample/get_camera_intrinsics.py --type realsense --list
  python sample/get_camera_intrinsics.py --type realsense --width 640 --height 480
  python sample/get_camera_intrinsics.py --type realsense --stream depth --width 848 --height 480

  # --- OAK-D ---
  python sample/get_camera_intrinsics.py --type oakd --list
  python sample/get_camera_intrinsics.py --type oakd --socket rgb --width 1920 --height 1080
  python sample/get_camera_intrinsics.py --type oakd --socket left --width 1280 --height 800

  # --- Orbbec ---
  python sample/get_camera_intrinsics.py --type orbbec --list
  python sample/get_camera_intrinsics.py --type orbbec --width 640 --height 480
  python sample/get_camera_intrinsics.py --type orbbec --stream depth --width 640 --height 576

  # --- OpenCV (approximate) ---
  python sample/get_camera_intrinsics.py --type opencv --camera-id 0 --list
  python sample/get_camera_intrinsics.py --type opencv --camera-id 0 --width 640 --height 480
  python sample/get_camera_intrinsics.py --type opencv --camera-id 0 --width 1920 --height 1080 --hfov 78.0
"""

import argparse
import sys
from collections import defaultdict

import numpy as np


# ---------------------------------------------------------------------------
# RealSense
# ---------------------------------------------------------------------------

def _require_realsense():
    try:
        import pyrealsense2 as rs
        return rs
    except ImportError:
        print("Error: pyrealsense2 is not installed.")
        print("  Install with:  pip install pyrealsense2")
        sys.exit(1)


def list_realsense_devices():
    """Print all connected RealSense devices and their stream profiles."""
    rs = _require_realsense()
    ctx = rs.context()
    devices = ctx.query_devices()

    if len(devices) == 0:
        print("No RealSense device found.")
        return

    for dev in devices:
        name   = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        fw     = dev.get_info(rs.camera_info.firmware_version)
        print(f"\nDevice : {name}")
        print(f"  Serial  : {serial}")
        print(f"  Firmware: {fw}")

        for sensor in dev.query_sensors():
            sensor_name = sensor.get_info(rs.camera_info.name)
            profiles = [
                p for p in sensor.get_stream_profiles()
                if p.is_video_stream_profile()
            ]
            if not profiles:
                continue

            print(f"\n  Sensor: {sensor_name}")

            # Group: (stream_name, format, width, height) → set of fps values
            grouped = defaultdict(set)
            for p in profiles:
                vp = p.as_video_stream_profile()
                key = (p.stream_name(), str(p.format()), vp.width(), vp.height())
                grouped[key].add(p.fps())

            for (stream_name, fmt, w, h), fps_set in sorted(grouped.items()):
                fps_str = ", ".join(str(f) for f in sorted(fps_set))
                print(f"    [{stream_name}]  {w:4d} x {h:4d}  @ {fps_str} fps  (format={fmt})")


def query_realsense(width: int, height: int, stream: str, fps: int) -> dict:
    """Query calibrated intrinsics from RealSense firmware."""
    rs = _require_realsense()

    stream_map = {
        "color":    (rs.stream.color,    rs.format.bgr8),
        "depth":    (rs.stream.depth,    rs.format.z16),
        "infrared": (rs.stream.infrared, rs.format.y8),
    }
    if stream not in stream_map:
        print(f"Error: --stream must be one of {list(stream_map)} for RealSense.")
        sys.exit(1)

    rs_stream, rs_format = stream_map[stream]
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs_stream, width, height, rs_format, fps)

    try:
        profile = pipeline.start(config)
    except RuntimeError as e:
        print(f"Error: Failed to start RealSense pipeline: {e}")
        print("  → Check camera connection and use --list to see supported modes.")
        sys.exit(1)

    try:
        sp   = profile.get_stream(rs_stream).as_video_stream_profile()
        intr = sp.get_intrinsics()
        return {
            "fx":          intr.fx,
            "fy":          intr.fy,
            "cx":          intr.ppx,
            "cy":          intr.ppy,
            "width":       intr.width,
            "height":      intr.height,
            "dist_model":  str(intr.model),
            "dist_coeffs": intr.coeffs,
        }
    finally:
        pipeline.stop()


# ---------------------------------------------------------------------------
# OAK-D (Luxonis DepthAI)
# ---------------------------------------------------------------------------

def _require_depthai():
    try:
        import depthai as dai
        return dai
    except ImportError:
        print("Error: depthai is not installed.")
        print("  Install with:  pip install depthai")
        sys.exit(1)


# Canonical socket aliases understood by the CLI
_OAKD_SOCKET_ALIASES = {
    "rgb":   "RGB",
    "cam_a": "CAM_A",
    "left":  "LEFT",
    "cam_b": "CAM_B",
    "right": "RIGHT",
    "cam_c": "CAM_C",
    "cam_d": "CAM_D",
}


def _oakd_socket(dai, name: str):
    """Resolve a socket alias string to a dai.CameraBoardSocket enum value."""
    attr = _OAKD_SOCKET_ALIASES.get(name.lower())
    if attr is None:
        print(f"Error: Unknown --socket '{name}'. "
              f"Choose from: {list(_OAKD_SOCKET_ALIASES)}")
        sys.exit(1)
    try:
        return getattr(dai.CameraBoardSocket, attr)
    except AttributeError:
        print(f"Error: Socket '{attr}' is not supported by this depthai version.")
        sys.exit(1)


def list_oakd_devices():
    """List connected OAK-D devices, their cameras, and calibration status."""
    dai = _require_depthai()
    devices = dai.Device.getAllAvailableDevices()

    if not devices:
        print("No OAK-D device found.")
        return

    for dev_info in devices:
        print(f"\nDevice : {dev_info.name}")
        print(f"  MxId   : {dev_info.getMxId()}")

        try:
            with dai.Device(dev_info) as device:
                cameras = device.getConnectedCameraFeatures()
                calib   = device.readCalibration()

                print("  Cameras (use --socket <name> to select):")
                for cam in cameras:
                    socket_name = cam.socket.name.lower()
                    sensor      = cam.sensorName.strip('\x00') or "unknown"

                    # Try to get native calibration resolution
                    try:
                        M  = calib.getCameraIntrinsics(cam.socket)
                        fx = M[0][0]
                        fy = M[1][1]
                        calib_info = f"calibrated  (native fx={fx:.1f} fy={fy:.1f})"
                    except Exception:
                        calib_info = "no calibration data"

                    print(f"    --socket {socket_name:<8}  sensor={sensor:<20}  {calib_info}")

        except Exception as e:
            print(f"  Error reading device: {e}")


def query_oakd(socket_name: str, width: int, height: int, device_id: str) -> dict:
    """Read calibrated intrinsics from OAK-D firmware, scaled to requested resolution."""
    dai = _require_depthai()
    socket = _oakd_socket(dai, socket_name)

    try:
        if device_id:
            device_ctx = dai.Device(dai.DeviceInfo(device_id))
        else:
            device_ctx = dai.Device()
    except Exception as e:
        print(f"Error: Failed to open OAK-D device: {e}")
        sys.exit(1)

    with device_ctx as device:
        try:
            calib = device.readCalibration()
        except Exception as e:
            print(f"Error: Failed to read calibration: {e}")
            sys.exit(1)

        try:
            # getCameraIntrinsics scales the stored calibration to the given resolution
            M = calib.getCameraIntrinsics(socket, width, height)
        except Exception as e:
            print(f"Error: Cannot get intrinsics for socket '{socket_name}' "
                  f"at {width}x{height}: {e}")
            print("  → Use --list to check available sockets.")
            sys.exit(1)

        try:
            dist_coeffs = calib.getDistortionCoefficients(socket)
        except Exception:
            dist_coeffs = []

        try:
            dist_model = str(calib.getDistortionModel(socket))
        except Exception:
            dist_model = ""

        return {
            "fx":          M[0][0],
            "fy":          M[1][1],
            "cx":          M[0][2],
            "cy":          M[1][2],
            "width":       width,
            "height":      height,
            "dist_model":  dist_model,
            "dist_coeffs": dist_coeffs,
        }


# ---------------------------------------------------------------------------
# Orbbec (pyorbbecsdk)
# ---------------------------------------------------------------------------

def _require_orbbec():
    try:
        import pyorbbecsdk as ob
        return ob
    except ImportError:
        print("Error: pyorbbecsdk is not installed.")
        print("  See: https://github.com/orbbec/pyorbbecsdk")
        sys.exit(1)


def list_orbbec_devices():
    """List connected Orbbec devices and their supported stream profiles."""
    ob  = _require_orbbec()
    ctx = ob.Context()
    dev_list = ctx.query_devices()
    count    = dev_list.get_count()

    if count == 0:
        print("No Orbbec device found.")
        return

    sensor_type_names = {
        ob.OBSensorType.OB_SENSOR_COLOR:    "Color",
        ob.OBSensorType.OB_SENSOR_DEPTH:    "Depth",
        ob.OBSensorType.OB_SENSOR_IR:       "IR",
        ob.OBSensorType.OB_SENSOR_IR_LEFT:  "IR Left",
        ob.OBSensorType.OB_SENSOR_IR_RIGHT: "IR Right",
    }

    for i in range(count):
        device = dev_list.get_device(i)
        info   = device.get_device_info()
        print(f"\nDevice [{i}]: {info.get_name()}")
        print(f"  Serial  : {info.get_serial_number()}")
        print(f"  Firmware: {info.get_firmware_version()}")

        sensor_list = device.get_sensor_list()
        for j in range(sensor_list.get_count()):
            sensor      = sensor_list.get_sensor_by_index(j)
            stype       = sensor.get_type()
            sname       = sensor_type_names.get(stype, str(stype))

            try:
                profile_list = sensor.get_stream_profile_list()
            except Exception:
                continue

            # Group: (width, height, format) → set of fps values
            grouped = defaultdict(set)
            for k in range(profile_list.get_count()):
                try:
                    profile = profile_list.get_stream_profile_by_index(k)
                    vp      = profile.as_video_stream_profile()
                    grouped[(vp.get_width(), vp.get_height(),
                             str(vp.get_format()))].add(vp.get_fps())
                except Exception:
                    continue

            if not grouped:
                continue

            print(f"\n  Sensor: {sname}")
            for (w, h, fmt), fps_set in sorted(grouped.items()):
                fps_str = ", ".join(str(f) for f in sorted(fps_set))
                print(f"    {w:4d} x {h:4d}  @ {fps_str} fps  (format={fmt})")


def query_orbbec(camera_id: int, width: int, height: int,
                 stream: str, fps: int) -> dict:
    """Start an Orbbec pipeline briefly to read calibrated intrinsics."""
    ob = _require_orbbec()

    stream_to_sensor = {
        "color": ob.OBSensorType.OB_SENSOR_COLOR,
        "depth": ob.OBSensorType.OB_SENSOR_DEPTH,
        "ir":    ob.OBSensorType.OB_SENSOR_IR,
    }
    if stream not in stream_to_sensor:
        print(f"Error: --stream must be one of {list(stream_to_sensor)} for Orbbec.")
        sys.exit(1)

    sensor_type = stream_to_sensor[stream]

    ctx      = ob.Context()
    dev_list = ctx.query_devices()
    if camera_id >= dev_list.get_count():
        print(f"Error: Device index {camera_id} not found "
              f"({dev_list.get_count()} device(s) connected).")
        sys.exit(1)

    device   = dev_list.get_device(camera_id)
    pipeline = ob.Pipeline(device)
    config   = ob.Config()

    try:
        profile_list = pipeline.get_stream_profile_list(sensor_type)
    except Exception as e:
        print(f"Error: Stream '{stream}' not available on this device: {e}")
        sys.exit(1)

    # Try to find matching profile; fall back to any fps if exact fps not found
    profile = None
    for try_fps in (fps, 0):
        try:
            profile = profile_list.get_video_stream_profile(
                width, height, ob.OBFormat.OB_FORMAT_UNKNOWN, try_fps)
            break
        except Exception:
            continue

    if profile is None:
        print(f"Error: No profile found for {width}x{height} on '{stream}' stream.")
        print("  → Use --list to see supported resolutions.")
        sys.exit(1)

    config.enable_stream(profile)

    try:
        pipeline.start(config)
        cam_param = pipeline.get_camera_param()
    except Exception as e:
        print(f"Error: Failed to start pipeline or get camera params: {e}")
        sys.exit(1)
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass

    if stream == "color":
        intr = cam_param.rgb_intrinsic
        dist = cam_param.rgb_distortion
    else:
        intr = cam_param.depth_intrinsic
        dist = cam_param.depth_distortion

    dist_coeffs = [dist.k1, dist.k2, dist.p1, dist.p2,
                   dist.k3, dist.k4, dist.k5, dist.k6]

    return {
        "fx":          intr.fx,
        "fy":          intr.fy,
        "cx":          intr.cx,
        "cy":          intr.cy,
        "width":       intr.width,
        "height":      intr.height,
        "dist_model":  "Brown-Conrady",
        "dist_coeffs": dist_coeffs,
    }


# ---------------------------------------------------------------------------
# OpenCV (generic webcam – approximate intrinsics)
# ---------------------------------------------------------------------------

_PROBE_RESOLUTIONS = [
    (320,  240), (424,  240), (640,  360), (640,  480),
    (848,  480), (960,  540), (1280, 720), (1920, 1080),
    (2560, 1440), (3840, 2160),
]


def list_opencv_resolutions(camera_id: int):
    """Probe common resolutions on an OpenCV camera and print supported ones."""
    import cv2
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Cannot open camera {camera_id}")
        sys.exit(1)

    print(f"\nCamera {camera_id} — supported resolutions (probe):")
    seen = []
    for w, h in _PROBE_RESOLUTIONS:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (aw, ah) not in seen:
            seen.append((aw, ah))
            print(f"  {aw:4d} x {ah:4d}")
    cap.release()


def query_opencv_resolution(camera_id: int, width: int, height: int):
    """Open camera, request resolution, return what the driver reports."""
    import cv2
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Cannot open camera {camera_id}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return actual_w, actual_h


def approximate_intrinsics(width: int, height: int, hfov_deg: float):
    """Approximate intrinsics from resolution and horizontal FOV (square pixels assumed)."""
    hfov_rad = np.radians(hfov_deg)
    fx = (width / 2.0) / np.tan(hfov_rad / 2.0)
    cx = (width  - 1) / 2.0
    cy = (height - 1) / 2.0
    return fx, fx, cx, cy  # fy = fx


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_intrinsics(
    fx: float, fy: float, cx: float, cy: float,
    width: int, height: int,
    source: str = "", dist_model: str = "", dist_coeffs=None, note: str = "",
):
    print()
    if source:
        print(f"Source    : {source}")
    print(f"Resolution: {width} x {height}")
    print()
    print("Intrinsic matrix K:")
    print(f"  [ {fx:10.4f}      0.0000  {cx:10.4f} ]")
    print(f"  [      0.0000  {fy:10.4f}  {cy:10.4f} ]")
    print( "  [      0.0000      0.0000       1.0000 ]")
    print()
    if dist_model:
        print(f"Distortion model : {dist_model}")
    if dist_coeffs:
        coeffs_str = "  ".join(f"{c:.6f}" for c in dist_coeffs)
        print(f"Distortion coeffs: [{coeffs_str}]")
    print()
    if note:
        print(f"Note: {note}")
        print()
    print("CLI arguments for visual_odometry.py:")
    print(f"  --fx {fx:.4f} --fy {fy:.4f} --cx {cx:.4f} --cy {cy:.4f}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Query or estimate camera intrinsic parameters.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--type", "-t",
        choices=["realsense", "oakd", "orbbec", "opencv"],
        required=True,
        help="Camera type",
    )
    parser.add_argument(
        "--width", "-W",
        type=int, default=640,
        help="Image width in pixels (default: 640)",
    )
    parser.add_argument(
        "--height", "-H",
        type=int, default=480,
        help="Image height in pixels (default: 480)",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List connected devices / supported resolutions and exit",
    )

    # RealSense / Orbbec shared options
    ro_group = parser.add_argument_group("RealSense / Orbbec options")
    ro_group.add_argument(
        "--stream",
        default="color",
        metavar="STREAM",
        help=(
            "Stream type to query (default: color). "
            "RealSense: color | depth | infrared. "
            "Orbbec:    color | depth | ir."
        ),
    )
    ro_group.add_argument(
        "--fps",
        type=int, default=30,
        help="Frame rate (default: 30)",
    )

    # OAK-D options
    oak_group = parser.add_argument_group("OAK-D options")
    oak_group.add_argument(
        "--socket",
        default="rgb",
        metavar="SOCKET",
        help=(
            "Camera socket to query (default: rgb). "
            f"Choices: {list(_OAKD_SOCKET_ALIASES)}"
        ),
    )
    oak_group.add_argument(
        "--device-id",
        default="",
        metavar="MXID",
        help="OAK-D device MxId (default: first found device)",
    )

    # OpenCV / Orbbec device index
    cv_group = parser.add_argument_group("OpenCV / Orbbec options")
    cv_group.add_argument(
        "--camera-id",
        type=int, default=0,
        help="Camera / device index (default: 0)",
    )
    cv_group.add_argument(
        "--hfov",
        type=float, default=69.0,
        help=(
            "Horizontal FOV in degrees for approximate OpenCV intrinsics "
            "(default: 69.0). Common values: webcam ≈ 69°, wide-angle ≈ 90°."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.type == "realsense":
        if args.list:
            list_realsense_devices()
        else:
            result = query_realsense(args.width, args.height, args.stream, args.fps)
            print_intrinsics(
                fx=result["fx"], fy=result["fy"],
                cx=result["cx"], cy=result["cy"],
                width=result["width"], height=result["height"],
                source=f"Intel RealSense — {args.stream} stream",
                dist_model=result["dist_model"],
                dist_coeffs=result["dist_coeffs"],
            )

    elif args.type == "oakd":
        if args.list:
            list_oakd_devices()
        else:
            result = query_oakd(args.socket, args.width, args.height, args.device_id)
            print_intrinsics(
                fx=result["fx"], fy=result["fy"],
                cx=result["cx"], cy=result["cy"],
                width=result["width"], height=result["height"],
                source=f"OAK-D — socket={args.socket}",
                dist_model=result["dist_model"],
                dist_coeffs=result["dist_coeffs"],
            )

    elif args.type == "orbbec":
        if args.list:
            list_orbbec_devices()
        else:
            result = query_orbbec(
                args.camera_id, args.width, args.height, args.stream, args.fps)
            print_intrinsics(
                fx=result["fx"], fy=result["fy"],
                cx=result["cx"], cy=result["cy"],
                width=result["width"], height=result["height"],
                source=f"Orbbec [{args.camera_id}] — {args.stream} stream",
                dist_model=result["dist_model"],
                dist_coeffs=result["dist_coeffs"],
            )

    elif args.type == "opencv":
        if args.list:
            list_opencv_resolutions(args.camera_id)
        else:
            actual_w, actual_h = query_opencv_resolution(
                args.camera_id, args.width, args.height)
            note = ""
            if actual_w != args.width or actual_h != args.height:
                note = (f"Requested {args.width}x{args.height} but camera returned "
                        f"{actual_w}x{actual_h}. Intrinsics computed for actual size.  ")
            fx, fy, cx, cy = approximate_intrinsics(actual_w, actual_h, args.hfov)
            print_intrinsics(
                fx=fx, fy=fy, cx=cx, cy=cy,
                width=actual_w, height=actual_h,
                source=(f"OpenCV camera {args.camera_id} "
                        f"(approximate, hfov={args.hfov:.1f}°)"),
                note=(note +
                      "APPROXIMATE values from assumed FOV. "
                      "For accuracy, run OpenCV checkerboard calibration."),
            )


if __name__ == "__main__":
    main()
