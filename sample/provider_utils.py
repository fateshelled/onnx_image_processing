"""
ONNX Runtime execution provider utilities for image matching samples.

Provides helper functions to create ONNX Runtime sessions with different
execution providers (CPU, CUDA, TensorRT).
"""

import onnxruntime as ort


def get_provider_config(provider_name: str = "cuda", **kwargs) -> tuple:
    """
    Get ONNX Runtime execution provider configuration.

    Args:
        provider_name: Provider name - "cpu", "cuda", or "tensorrt"
        **kwargs: Provider-specific options to override defaults

    Returns:
        Provider configuration tuple (name, options_dict) or just name for CPU

    Example:
        >>> provider = get_provider_config("tensorrt", trt_fp16_enable=True)
        >>> session = ort.InferenceSession(model_path, providers=[provider])
    """
    provider_name = provider_name.lower()

    if provider_name == "cpu":
        return "CPUExecutionProvider"

    elif provider_name == "cuda":
        default_options = {}
        options = {**default_options, **kwargs}
        return ("CUDAExecutionProvider", options)

    elif provider_name == "tensorrt":
        default_options = {
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": ".",
            "trt_fp16_enable": False,  # Use FP32 to avoid NaN issues with Sinkhorn
            # onnxruntime>=1.21.0 breaking changes
            # https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#data-dependant-shape-dds-ops
            # TopK excluded due to Int64/Int32 index type mismatch
            "trt_op_types_to_exclude": "NonMaxSuppression,NonZero,RoiAlign,TopK",
        }
        options = {**default_options, **kwargs}
        return ("TensorrtExecutionProvider", options)

    else:
        raise ValueError(
            f"Unknown provider: {provider_name}. Choose from: cpu, cuda, tensorrt"
        )


def create_session(
    model_path: str, provider: str = "cuda", **provider_options
) -> ort.InferenceSession:
    """
    Create ONNX Runtime inference session with specified provider.

    Args:
        model_path: Path to ONNX model file
        provider: Provider name - "cpu", "cuda", or "tensorrt" (default: "tensorrt")
        **provider_options: Provider-specific options to override defaults

    Returns:
        ONNX Runtime InferenceSession

    Example:
        >>> session = create_session("model.onnx", "tensorrt", trt_fp16_enable=True)
        >>> session = create_session("model.onnx", "cpu")
    """
    ort.preload_dlls()
    provider_config = get_provider_config(provider, **provider_options)
    return ort.InferenceSession(model_path, providers=[provider_config])
