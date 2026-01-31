"""
Compute Backend Abstraction Layer

Provides a unified interface for array operations that can work with:
- CuPy (NVIDIA CUDA)
- MLX (Apple Silicon Metal)
- NumPy (CPU fallback)

This module abstracts away the underlying compute library and provides
NumPy-based implementations of CUDA kernels for cross-platform compatibility.
"""
import logging
import platform
from enum import Enum
from typing import Optional, Any, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)


class BackendType(Enum):
    """Available compute backend types."""
    CUPY = "cupy"
    MLX = "mlx"
    NUMPY = "numpy"


class ComputeBackend:
    """
    Unified compute backend that abstracts array operations.

    Automatically selects the best available backend:
    1. CuPy (if NVIDIA GPU available)
    2. MLX (if Apple Silicon)
    3. NumPy (fallback)
    """

    _instance = None
    _backend_type = None
    _xp = None  # Array library (cupy, mlx.core, or numpy)

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._detect_backend()

    def _detect_backend(self):
        """Detect and initialize the best available backend."""
        # Try CuPy first
        try:
            import cupy as cp
            # Test CUDA availability
            test = cp.array([1, 2, 3])
            _ = cp.sum(test)
            self._xp = cp
            self._backend_type = BackendType.CUPY
            logger.info("ComputeBackend: Using CuPy (NVIDIA CUDA)")
            return
        except (ImportError, Exception) as e:
            logger.debug(f"CuPy not available: {e}")

        # Try MLX on Apple Silicon
        is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
        if is_apple_silicon:
            try:
                import mlx.core as mx
                test = mx.array([1, 2, 3])
                _ = mx.sum(test)
                mx.eval(_)
                self._xp = mx
                self._backend_type = BackendType.MLX
                logger.info("ComputeBackend: Using MLX (Apple Silicon)")
                return
            except (ImportError, Exception) as e:
                logger.debug(f"MLX not available: {e}")

        # Fall back to NumPy
        self._xp = np
        self._backend_type = BackendType.NUMPY
        logger.info("ComputeBackend: Using NumPy (CPU)")

    @property
    def backend_type(self) -> BackendType:
        """Get the current backend type."""
        return self._backend_type

    @property
    def xp(self):
        """Get the array library (cupy, mlx.core, or numpy)."""
        return self._xp

    @property
    def is_gpu(self) -> bool:
        """Check if using GPU acceleration."""
        return self._backend_type in (BackendType.CUPY, BackendType.MLX)

    def array(self, data, dtype=None) -> Any:
        """Create array from data."""
        if self._backend_type == BackendType.MLX:
            import mlx.core as mx
            return mx.array(data, dtype=self._convert_dtype(dtype))
        return self._xp.array(data, dtype=dtype)

    def asarray(self, data, dtype=None) -> Any:
        """Convert data to array."""
        if self._backend_type == BackendType.MLX:
            import mlx.core as mx
            return mx.array(data, dtype=self._convert_dtype(dtype))
        return self._xp.asarray(data, dtype=dtype)

    def zeros(self, shape, dtype=None) -> Any:
        """Create zero-filled array."""
        if self._backend_type == BackendType.MLX:
            import mlx.core as mx
            return mx.zeros(shape, dtype=self._convert_dtype(dtype))
        return self._xp.zeros(shape, dtype=dtype)

    def ones(self, shape, dtype=None) -> Any:
        """Create one-filled array."""
        if self._backend_type == BackendType.MLX:
            import mlx.core as mx
            return mx.ones(shape, dtype=self._convert_dtype(dtype))
        return self._xp.ones(shape, dtype=dtype)

    def full(self, shape, fill_value, dtype=None) -> Any:
        """Create array filled with value."""
        if self._backend_type == BackendType.MLX:
            import mlx.core as mx
            return mx.full(shape, fill_value, dtype=self._convert_dtype(dtype))
        return self._xp.full(shape, fill_value, dtype=dtype)

    def empty(self, shape, dtype=None) -> Any:
        """Create uninitialized array."""
        if self._backend_type == BackendType.MLX:
            # MLX doesn't have empty, use zeros
            import mlx.core as mx
            return mx.zeros(shape, dtype=self._convert_dtype(dtype))
        return self._xp.empty(shape, dtype=dtype)

    def arange(self, *args, dtype=None) -> Any:
        """Create array with evenly spaced values."""
        if self._backend_type == BackendType.MLX:
            import mlx.core as mx
            return mx.arange(*args, dtype=self._convert_dtype(dtype))
        return self._xp.arange(*args, dtype=dtype)

    def copy(self, array) -> Any:
        """Copy array."""
        if self._backend_type == BackendType.MLX:
            import mlx.core as mx
            return mx.array(array)
        return self._xp.copy(array)

    def where(self, condition, x=None, y=None) -> Any:
        """Conditional array selection."""
        if x is None and y is None:
            return self._xp.where(condition)
        return self._xp.where(condition, x, y)

    def sum(self, array, axis=None) -> Any:
        """Sum array elements."""
        return self._xp.sum(array, axis=axis)

    def min(self, array, axis=None) -> Any:
        """Minimum of array elements."""
        return self._xp.min(array, axis=axis)

    def max(self, array, axis=None) -> Any:
        """Maximum of array elements."""
        return self._xp.max(array, axis=axis)

    def argmin(self, array, axis=None) -> Any:
        """Index of minimum element."""
        return self._xp.argmin(array, axis=axis)

    def argmax(self, array, axis=None) -> Any:
        """Index of maximum element."""
        return self._xp.argmax(array, axis=axis)

    def concatenate(self, arrays, axis=0) -> Any:
        """Concatenate arrays."""
        return self._xp.concatenate(arrays, axis=axis)

    def vstack(self, arrays) -> Any:
        """Stack arrays vertically."""
        if self._backend_type == BackendType.MLX:
            import mlx.core as mx
            return mx.concatenate(arrays, axis=0)
        return self._xp.vstack(arrays)

    def hstack(self, arrays) -> Any:
        """Stack arrays horizontally."""
        if self._backend_type == BackendType.MLX:
            import mlx.core as mx
            return mx.concatenate(arrays, axis=1)
        return self._xp.hstack(arrays)

    def unique(self, array) -> Any:
        """Find unique elements."""
        if self._backend_type == BackendType.MLX:
            # MLX doesn't have unique, convert to numpy
            return np.unique(self.to_numpy(array))
        return self._xp.unique(array)

    def to_numpy(self, array) -> np.ndarray:
        """Convert array to numpy."""
        if self._backend_type == BackendType.CUPY:
            if hasattr(array, 'get'):
                return array.get()
            return np.asarray(array)
        elif self._backend_type == BackendType.MLX:
            import mlx.core as mx
            if hasattr(array, '__class__') and 'mlx' in str(type(array)):
                mx.eval(array)
                return np.array(array)
            return np.asarray(array)
        return np.asarray(array)

    def from_numpy(self, array: np.ndarray) -> Any:
        """Convert numpy array to backend array."""
        if self._backend_type == BackendType.CUPY:
            return self._xp.asarray(array)
        elif self._backend_type == BackendType.MLX:
            import mlx.core as mx
            return mx.array(array)
        return array

    def synchronize(self):
        """Synchronize GPU operations."""
        if self._backend_type == BackendType.CUPY:
            import cupy as cp
            cp.cuda.Stream.null.synchronize()
        elif self._backend_type == BackendType.MLX:
            # MLX syncs on eval or when accessing values
            pass

    def _convert_dtype(self, dtype):
        """Convert numpy dtype to MLX dtype."""
        if dtype is None:
            return None

        if self._backend_type != BackendType.MLX:
            return dtype

        import mlx.core as mx

        dtype_map = {
            np.float32: mx.float32,
            np.float64: mx.float32,
            np.int32: mx.int32,
            np.int64: mx.int64,
            np.int16: mx.int16,
            np.int8: mx.int8,
            np.uint8: mx.uint8,
            np.uint16: mx.uint16,
            np.uint32: mx.uint32,
            np.uint64: mx.uint64,
            np.bool_: mx.bool_,
            'float32': mx.float32,
            'float64': mx.float32,
            'int32': mx.int32,
            'int64': mx.int64,
            'int16': mx.int16,
            'int8': mx.int8,
            'uint8': mx.uint8,
            'uint16': mx.uint16,
            'uint32': mx.uint32,
            'uint64': mx.uint64,
            'bool': mx.bool_,
        }

        if hasattr(dtype, 'type'):
            dtype = dtype.type

        return dtype_map.get(dtype, mx.float32)

    # Type aliases for compatibility
    @property
    def float32(self):
        if self._backend_type == BackendType.MLX:
            import mlx.core as mx
            return mx.float32
        return self._xp.float32

    @property
    def float64(self):
        if self._backend_type == BackendType.MLX:
            import mlx.core as mx
            return mx.float32  # MLX prefers float32
        return self._xp.float64

    @property
    def int32(self):
        if self._backend_type == BackendType.MLX:
            import mlx.core as mx
            return mx.int32
        return self._xp.int32

    @property
    def int64(self):
        if self._backend_type == BackendType.MLX:
            import mlx.core as mx
            return mx.int64
        return self._xp.int64

    @property
    def int16(self):
        if self._backend_type == BackendType.MLX:
            import mlx.core as mx
            return mx.int16
        return self._xp.int16

    @property
    def int8(self):
        if self._backend_type == BackendType.MLX:
            import mlx.core as mx
            return mx.int8
        return self._xp.int8

    @property
    def uint8(self):
        if self._backend_type == BackendType.MLX:
            import mlx.core as mx
            return mx.uint8
        return self._xp.uint8

    @property
    def uint16(self):
        if self._backend_type == BackendType.MLX:
            import mlx.core as mx
            return mx.uint16
        return self._xp.uint16

    @property
    def uint32(self):
        if self._backend_type == BackendType.MLX:
            import mlx.core as mx
            return mx.uint32
        return self._xp.uint32

    @property
    def bool_(self):
        if self._backend_type == BackendType.MLX:
            import mlx.core as mx
            return mx.bool_
        return self._xp.bool_


# Global singleton instance
_backend = None


def get_backend() -> ComputeBackend:
    """Get the global compute backend instance."""
    global _backend
    if _backend is None:
        _backend = ComputeBackend()
    return _backend


def get_array_module(array=None):
    """
    Get the array module for the given array or the default backend.

    This is a compatibility function similar to cupy.get_array_module().
    """
    if array is not None:
        # Check if it's a cupy array
        if hasattr(array, '__class__'):
            class_name = str(type(array))
            if 'cupy' in class_name:
                try:
                    import cupy as cp
                    return cp
                except ImportError:
                    pass
            elif 'mlx' in class_name:
                try:
                    import mlx.core as mx
                    return mx
                except ImportError:
                    pass

    return get_backend().xp
