"""
ONNX model optimization utilities.

Provides a function to optimize exported ONNX models using onnx-simplifier
with a fallback to onnxoptimizer.
"""

import onnx

try:
    import onnxsim
except ImportError:
    onnxsim = None

try:
    import onnxoptimizer
except ImportError:
    onnxoptimizer = None


def optimize_onnx_model(model_path: str) -> str:
    """Optimize an ONNX model file in-place.

    Attempts optimization in the following order:
    1. onnx-simplifier (onnxsim)
    2. onnxoptimizer (fallback if onnxsim fails)
    3. No optimization (fallback if both fail)

    Args:
        model_path: Path to the ONNX model file.

    Returns:
        A string indicating which optimization was applied:
        "onnxsim", "onnxoptimizer", or "none".
    """
    model = onnx.load(model_path)

    # Try onnx-simplifier first
    if onnxsim is not None:
        try:
            model_simplified, check = onnxsim.simplify(model)
            if check:
                onnx.save(model_simplified, model_path)
                return "onnxsim"
            else:
                raise RuntimeError("onnxsim simplify check failed")
        except Exception as e:
            print(f"  onnxsim failed ({e}), trying onnxoptimizer...")
    else:
        print("  onnxsim not installed, trying onnxoptimizer...")

    # Fallback to onnxoptimizer
    if onnxoptimizer is not None:
        try:
            model_optimized = onnxoptimizer.optimize(model)
            onnx.save(model_optimized, model_path)
            return "onnxoptimizer"
        except Exception as e:
            print(f"  onnxoptimizer failed ({e}), saving unoptimized model.")
    else:
        print("  onnxoptimizer not installed, saving unoptimized model.")

    return "none"
