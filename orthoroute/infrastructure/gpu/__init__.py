"""GPU infrastructure adapters."""
from .cuda_provider import CUDAProvider
from .cpu_fallback import CPUFallbackProvider, CPUProvider
from .mlx_provider import MLXProvider, MLX_AVAILABLE, IS_APPLE_SILICON
from .compute_backend import (
    ComputeBackend,
    BackendType,
    get_backend,
    get_array_module,
)
from .numpy_kernels import (
    NumpyDijkstraKernels,
    NumpyViaKernels,
    NumpyROIKernels,
    get_dijkstra_kernels,
    get_via_kernels,
    get_roi_kernels,
)
from .cupy_compat import (
    cp,
    CUDA_AVAILABLE,
    to_numpy,
    from_numpy,
)

__all__ = [
    # GPU Providers
    'CUDAProvider',
    'CPUFallbackProvider',
    'CPUProvider',
    'MLXProvider',
    'MLX_AVAILABLE',
    'IS_APPLE_SILICON',
    # Compute Backend
    'ComputeBackend',
    'BackendType',
    'get_backend',
    'get_array_module',
    # NumPy Kernels
    'NumpyDijkstraKernels',
    'NumpyViaKernels',
    'NumpyROIKernels',
    'get_dijkstra_kernels',
    'get_via_kernels',
    'get_roi_kernels',
    # CuPy Compatibility
    'cp',
    'CUDA_AVAILABLE',
    'to_numpy',
    'from_numpy',
]


def get_best_provider():
    """
    Get the best available GPU provider.

    Returns provider in order of preference:
    1. CUDAProvider (NVIDIA GPU)
    2. MLXProvider (Apple Silicon)
    3. CPUFallbackProvider (CPU)
    """
    # Try CUDA first
    cuda = CUDAProvider()
    if cuda.is_available():
        return cuda

    # Try MLX on Apple Silicon
    if IS_APPLE_SILICON and MLX_AVAILABLE:
        mlx = MLXProvider()
        if mlx.is_available():
            return mlx

    # Fall back to CPU
    return CPUFallbackProvider()