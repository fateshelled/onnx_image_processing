"""
ONNX model optimization utilities.

Provides a function to optimize exported ONNX models using onnx-simplifier
with a fallback to onnxoptimizer.
"""

import os

import onnx
import onnxoptimizer
import onnxsim


def _remove_data_file(model_path: str) -> None:
    """Remove the external data file (.data) associated with an ONNX model."""
    data_path = model_path + ".data"
    if os.path.exists(data_path):
        os.remove(data_path)


def remove_external_data(model_path: str) -> None:
    """Load an ONNX model and re-save it with all tensor data internalized.

    This removes any external .data files generated during export.

    Args:
        model_path: Path to the ONNX model file.
    """
    model = onnx.load(model_path)
    onnx.save(model, model_path)
    _remove_data_file(model_path)


def optimize_onnx_model(model_path: str) -> str:
    """Optimize an ONNX model file in-place.

    Attempts optimization in the following order:
    1. onnx-simplifier (onnxsim)
    2. onnxoptimizer (fallback if onnxsim fails)
    3. No optimization (fallback if both fail)

    Any external data files (.data) are removed after saving.

    Args:
        model_path: Path to the ONNX model file.

    Returns:
        A string indicating which optimization was applied:
        "onnxsim", "onnxoptimizer", or "none".
    """
    model = onnx.load(model_path)

    # Try onnx-simplifier first
    try:
        model_simplified, check = onnxsim.simplify(model)
        if check:
            onnx.save(model_simplified, model_path)
            _remove_data_file(model_path)
            return "onnxsim"
        else:
            raise RuntimeError("onnxsim simplify check failed")
    except Exception as e:
        print(f"  onnxsim failed ({e}), trying onnxoptimizer...")

    # Fallback to onnxoptimizer
    try:
        model_optimized = onnxoptimizer.optimize(model)
        onnx.save(model_optimized, model_path)
        _remove_data_file(model_path)
        return "onnxoptimizer"
    except Exception as e:
        print(f"  onnxoptimizer failed ({e}), saving unoptimized model.")

    # Both optimizers failed; re-save to internalize any external data
    onnx.save(model, model_path)
    _remove_data_file(model_path)
    return "none"
