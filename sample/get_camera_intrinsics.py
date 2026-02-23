#!/usr/bin/env python3
"""
Camera intrinsic parameter query tool.

Outputs fx, fy, cx, cy for use with visual_odometry.py and other tools.

Supported camera types
  realsense  : Intel RealSense (D400 / L500 series) via pyrealsense2.
               Reads calibrated intrinsics directly from the device firmware.
  opencv     : Generic USB/built-in webcam via OpenCV.
               Computes approximate intrinsics from resolution + assumed FOV.

Usage examples
  # List connected RealSense devices and supported resolutions
  python sample/get_camera_intrinsics.py --type realsense --list

  # RealSense D435 color stream at 640x480
  python sample/get_camera_intrinsics.py --type realsense --width 640 --height 480

  # RealSense depth or infrared stream
  python sample/get_camera_intrinsics.py --type realsense --stream depth --width 848 --height 480

  # Generic webcam (approximate; assumes 69° horizontal FOV)
  python sample/get_camera_intrinsics.py --type opencv --camera-id 0 --width 640 --height 480

  # Generic webcam with known FOV
  python sample/get_camera_intrinsics.py --type opencv --camera-id 0 --width 1920 --height 1080 --hfov 78.0

  # List resolutions supported by OpenCV camera
  python sample/get_camera_intrinsics.py --type opencv --camera-id 0 --list
"""

import argparse
import sys

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

            # Group by stream type for compact display
            from collections import defaultdict
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
        print(f"Error: Unknown stream '{stream}'. Choose from: {list(stream_map)}")
        sys.exit(1)

    rs_stream, rs_format = stream_map[stream]
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs_stream, width, height, rs_format, fps)

    try:
        profile = pipeline.start(config)
    except RuntimeError as e:
        print(f"Error: Failed to start RealSense pipeline: {e}")
        print("  → Check that the camera is connected and the resolution/fps is supported.")
        print("  → Run with --list to see supported modes.")
        sys.exit(1)

    try:
        sp   = profile.get_stream(rs_stream).as_video_stream_profile()
        intr = sp.get_intrinsics()
        return {
            "fx":            intr.fx,
            "fy":            intr.fy,
            "cx":            intr.ppx,
            "cy":            intr.ppy,
            "width":         intr.width,
            "height":        intr.height,
            "dist_model":    str(intr.model),
            "dist_coeffs":   intr.coeffs,
        }
    finally:
        pipeline.stop()


# ---------------------------------------------------------------------------
# OpenCV (generic webcam)
# ---------------------------------------------------------------------------

# Common resolutions to probe when listing supported modes
_PROBE_RESOLUTIONS = [
    (320,  240),
    (424,  240),
    (640,  360),
    (640,  480),
    (848,  480),
    (960,  540),
    (1280, 720),
    (1920, 1080),
    (2560, 1440),
    (3840, 2160),
]


def list_opencv_resolutions(camera_id: int):
    """Probe common resolutions on an OpenCV camera and print supported ones."""
    import cv2
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Cannot open camera {camera_id}")
        sys.exit(1)

    print(f"\nCamera {camera_id} — supported resolutions (approximate probe):")
    supported = []
    for w, h in _PROBE_RESOLUTIONS:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (aw, ah) not in supported:
            supported.append((aw, ah))
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
    """
    Compute approximate intrinsics from image resolution and horizontal FOV.

    Assumes square pixels (fx == fy) and principal point at image centre.
    """
    hfov_rad = np.radians(hfov_deg)
    fx = (width / 2.0) / np.tan(hfov_rad / 2.0)
    fy = fx
    cx = (width  - 1) / 2.0
    cy = (height - 1) / 2.0
    return fx, fy, cx, cy


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_intrinsics(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    source: str = "",
    dist_model: str = "",
    dist_coeffs=None,
    note: str = "",
):
    print()
    if source:
        print(f"Source    : {source}")
    print(f"Resolution: {width} x {height}")
    print()
    print("Intrinsic matrix K:")
    print(f"  [ {fx:10.4f}      0.0000  {cx:10.4f} ]")
    print(f"  [      0.0000  {fy:10.4f}  {cy:10.4f} ]")
    print(f"  [      0.0000      0.0000       1.0000 ]")
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
        choices=["realsense", "opencv"],
        required=True,
        help="Camera type",
    )
    parser.add_argument(
        "--width", "-W",
        type=int,
        default=640,
        help="Image width in pixels (default: 640)",
    )
    parser.add_argument(
        "--height", "-H",
        type=int,
        default=480,
        help="Image height in pixels (default: 480)",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List connected devices / supported resolutions and exit",
    )

    # RealSense options
    rs_group = parser.add_argument_group("RealSense options")
    rs_group.add_argument(
        "--stream",
        choices=["color", "depth", "infrared"],
        default="color",
        help="Stream type to query (default: color)",
    )
    rs_group.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frame rate (default: 30)",
    )

    # OpenCV options
    cv_group = parser.add_argument_group("OpenCV options")
    cv_group.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="Camera device index (default: 0)",
    )
    cv_group.add_argument(
        "--hfov",
        type=float,
        default=69.0,
        help=(
            "Horizontal field-of-view in degrees used to compute approximate "
            "intrinsics (default: 69.0). "
            "Common values: built-in webcam ≈ 69°, wide-angle ≈ 90°, "
            "telephoto ≈ 45°."
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
                fx=result["fx"],
                fy=result["fy"],
                cx=result["cx"],
                cy=result["cy"],
                width=result["width"],
                height=result["height"],
                source=f"Intel RealSense — {args.stream} stream",
                dist_model=result["dist_model"],
                dist_coeffs=result["dist_coeffs"],
            )

    elif args.type == "opencv":
        if args.list:
            list_opencv_resolutions(args.camera_id)
        else:
            actual_w, actual_h = query_opencv_resolution(
                args.camera_id, args.width, args.height
            )
            note = ""
            if actual_w != args.width or actual_h != args.height:
                note = (
                    f"Requested {args.width}x{args.height} but camera "
                    f"returned {actual_w}x{actual_h}. "
                    f"Intrinsics are computed for the actual resolution."
                )
            fx, fy, cx, cy = approximate_intrinsics(actual_w, actual_h, args.hfov)
            print_intrinsics(
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                width=actual_w,
                height=actual_h,
                source=(
                    f"OpenCV camera {args.camera_id} "
                    f"(approximate, hfov={args.hfov:.1f}°)"
                ),
                note=(
                    (note + "  " if note else "")
                    + "These are APPROXIMATE values computed from the assumed FOV. "
                    "For accurate intrinsics run OpenCV camera calibration "
                    "(e.g., opencv_interactive-calibration or a checkerboard script)."
                ),
            )


if __name__ == "__main__":
    main()
