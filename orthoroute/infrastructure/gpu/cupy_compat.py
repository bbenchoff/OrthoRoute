"""
CuPy Compatibility Layer

This module provides a compatibility layer that allows existing code
using CuPy to work with NumPy or MLX when CuPy is not available.

Usage:
    Instead of:
        import cupy as cp

    Use:
        from orthoroute.infrastructure.gpu.cupy_compat import cp, CUDA_AVAILABLE

The 'cp' object will be CuPy when available, otherwise a NumPy-based
fallback that provides compatible operations.
"""
import logging
import platform
import numpy as np

logger = logging.getLogger(__name__)

# Try to import CuPy
CUDA_AVAILABLE = False
MLX_AVAILABLE = False

try:
    import cupy as _cupy
    # Test that CUDA actually works
    _test = _cupy.array([1, 2, 3])
    _ = _cupy.sum(_test)
    CUDA_AVAILABLE = True
    cp = _cupy
    logger.debug("CuPy/CUDA available")
except (ImportError, Exception) as e:
    logger.debug(f"CuPy not available: {e}")
    _cupy = None

# Try MLX on Apple Silicon
if not CUDA_AVAILABLE:
    IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"
    if IS_APPLE_SILICON:
        try:
            import mlx.core as mx
            _test = mx.array([1, 2, 3])
            _ = mx.sum(_test)
            mx.eval(_)
            MLX_AVAILABLE = True
            logger.debug("MLX available on Apple Silicon")
        except (ImportError, Exception) as e:
            logger.debug(f"MLX not available: {e}")


class NumpyArrayWrapper:
    """
    Wrapper that makes NumPy arrays behave more like CuPy arrays.

    This adds a .get() method that returns the array itself,
    allowing code that expects CuPy arrays to work with NumPy.
    """
    def __init__(self, array):
        self._array = np.asarray(array)

    def get(self):
        """Return the underlying NumPy array (CuPy compatibility)."""
        return self._array

    def __array__(self, dtype=None):
        if dtype is not None:
            return self._array.astype(dtype)
        return self._array

    def __getattr__(self, name):
        return getattr(self._array, name)

    def __getitem__(self, key):
        result = self._array[key]
        if isinstance(result, np.ndarray):
            return NumpyArrayWrapper(result)
        return result

    def __setitem__(self, key, value):
        if isinstance(value, NumpyArrayWrapper):
            self._array[key] = value._array
        else:
            self._array[key] = value

    def __len__(self):
        return len(self._array)

    def __repr__(self):
        return f"NumpyArrayWrapper({self._array!r})"


class NumpyCuPyCompat:
    """
    NumPy-based compatibility layer that mimics CuPy's API.

    This allows existing CuPy code to run on systems without CUDA
    by providing NumPy implementations of CuPy functions.
    """

    # Array type for isinstance checks
    ndarray = np.ndarray

    # Data types
    float32 = np.float32
    float64 = np.float64
    int32 = np.int32
    int64 = np.int64
    int16 = np.int16
    int8 = np.int8
    uint8 = np.uint8
    uint16 = np.uint16
    uint32 = np.uint32
    uint64 = np.uint64
    bool_ = np.bool_
    inf = np.inf

    def __init__(self):
        self._using_numpy = True

    def array(self, data, dtype=None):
        """Create array from data."""
        return np.array(data, dtype=dtype)

    def asarray(self, data, dtype=None):
        """Convert to array."""
        if hasattr(data, 'get'):
            data = data.get()
        return np.asarray(data, dtype=dtype)

    def zeros(self, shape, dtype=None):
        """Create zero-filled array."""
        return np.zeros(shape, dtype=dtype or np.float32)

    def ones(self, shape, dtype=None):
        """Create one-filled array."""
        return np.ones(shape, dtype=dtype or np.float32)

    def full(self, shape, fill_value, dtype=None):
        """Create array filled with value."""
        return np.full(shape, fill_value, dtype=dtype or np.float32)

    def empty(self, shape, dtype=None):
        """Create uninitialized array."""
        return np.empty(shape, dtype=dtype or np.float32)

    def arange(self, *args, dtype=None):
        """Create array with evenly spaced values."""
        return np.arange(*args, dtype=dtype)

    def copy(self, array):
        """Copy array."""
        if hasattr(array, 'get'):
            array = array.get()
        return np.copy(array)

    def where(self, condition, x=None, y=None):
        """Conditional array selection."""
        if hasattr(condition, 'get'):
            condition = condition.get()
        if x is not None and hasattr(x, 'get'):
            x = x.get()
        if y is not None and hasattr(y, 'get'):
            y = y.get()
        if x is None and y is None:
            return np.where(condition)
        return np.where(condition, x, y)

    def sum(self, array, axis=None):
        """Sum array elements."""
        if hasattr(array, 'get'):
            array = array.get()
        return np.sum(array, axis=axis)

    def min(self, array, axis=None):
        """Minimum of array elements."""
        if hasattr(array, 'get'):
            array = array.get()
        return np.min(array, axis=axis)

    def max(self, array, axis=None):
        """Maximum of array elements."""
        if hasattr(array, 'get'):
            array = array.get()
        return np.max(array, axis=axis)

    def argmin(self, array, axis=None):
        """Index of minimum element."""
        if hasattr(array, 'get'):
            array = array.get()
        return np.argmin(array, axis=axis)

    def argmax(self, array, axis=None):
        """Index of maximum element."""
        if hasattr(array, 'get'):
            array = array.get()
        return np.argmax(array, axis=axis)

    def concatenate(self, arrays, axis=0):
        """Concatenate arrays."""
        arrays = [a.get() if hasattr(a, 'get') else a for a in arrays]
        return np.concatenate(arrays, axis=axis)

    def vstack(self, arrays):
        """Stack arrays vertically."""
        arrays = [a.get() if hasattr(a, 'get') else a for a in arrays]
        return np.vstack(arrays)

    def hstack(self, arrays):
        """Stack arrays horizontally."""
        arrays = [a.get() if hasattr(a, 'get') else a for a in arrays]
        return np.hstack(arrays)

    def unique(self, array):
        """Find unique elements."""
        if hasattr(array, 'get'):
            array = array.get()
        return np.unique(array)

    def sqrt(self, array):
        """Square root."""
        if hasattr(array, 'get'):
            array = array.get()
        return np.sqrt(array)

    def abs(self, array):
        """Absolute value."""
        if hasattr(array, 'get'):
            array = array.get()
        return np.abs(array)

    def maximum(self, a, b):
        """Element-wise maximum."""
        if hasattr(a, 'get'):
            a = a.get()
        if hasattr(b, 'get'):
            b = b.get()
        return np.maximum(a, b)

    def minimum(self, a, b):
        """Element-wise minimum."""
        if hasattr(a, 'get'):
            a = a.get()
        if hasattr(b, 'get'):
            b = b.get()
        return np.minimum(a, b)

    def get_default_memory_pool(self):
        """Get memory pool (no-op for NumPy)."""
        return _NumpyMemoryPool()

    class cuda:
        """CUDA namespace stub for NumPy fallback."""

        class Device:
            """Dummy CUDA device."""
            def __init__(self, device_id=0):
                self.id = device_id
                self.compute_capability = (0, 0)
                self.attributes = {'Name': 'CPU (NumPy fallback)'}

        class Stream:
            """Dummy CUDA stream."""
            null = None

            def __init__(self):
                pass

            def synchronize(self):
                pass

            @classmethod
            def _init_null(cls):
                if cls.null is None:
                    cls.null = cls()

        class Event:
            """Dummy CUDA event."""
            def __init__(self):
                import time
                self._time = time.time()

            def record(self):
                import time
                self._time = time.time()

            def synchronize(self):
                pass

        @staticmethod
        def get_elapsed_time(start, end):
            """Get elapsed time between events in milliseconds."""
            return (end._time - start._time) * 1000.0

        class runtime:
            """CUDA runtime stub."""
            @staticmethod
            def memGetInfo():
                """Get memory info (returns system memory)."""
                try:
                    import psutil
                    vm = psutil.virtual_memory()
                    return (vm.available, vm.total)
                except ImportError:
                    return (8 * 1024**3, 16 * 1024**3)  # Default 8GB/16GB

    def RawKernel(self, code, name):
        """
        Create a RawKernel (stub that logs warning).

        CUDA kernels cannot run on CPU, so this returns a dummy
        that logs when called.
        """
        return _NumpyRawKernel(name)


# Initialize cuda.Stream.null
NumpyCuPyCompat.cuda.Stream._init_null()


class _NumpyMemoryPool:
    """Dummy memory pool for NumPy fallback."""

    def free_all_blocks(self):
        """No-op for NumPy."""
        pass

    def set_limit(self, size=None):
        """No-op for NumPy."""
        pass

    def used_bytes(self):
        """Return 0 for NumPy."""
        return 0

    def total_bytes(self):
        """Return 0 for NumPy."""
        return 0


class _NumpyRawKernel:
    """
    Stub for CUDA RawKernel when running on CPU.

    This logs a warning when called, as CUDA kernels cannot
    run on CPU. Code should check CUDA_AVAILABLE before
    attempting to use RawKernels.
    """

    def __init__(self, name):
        self.name = name
        self._warned = False

    def __call__(self, grid, block, args, **kwargs):
        if not self._warned:
            logger.warning(
                f"CUDA kernel '{self.name}' called but CUDA not available. "
                "Using NumPy fallback which may be slower."
            )
            self._warned = True
        # Return without doing anything - caller should have fallback path


# Export the appropriate 'cp' based on availability
if CUDA_AVAILABLE:
    cp = _cupy
    logger.info("Using CuPy for GPU acceleration")
else:
    cp = NumpyCuPyCompat()
    logger.info("Using NumPy fallback (CuPy not available)")


# Helper functions
def get_array_module(array=None):
    """
    Get the array module for the given array.

    Similar to cupy.get_array_module() but works with the fallback.
    """
    if CUDA_AVAILABLE:
        return _cupy.get_array_module(array)
    return np


def to_numpy(array):
    """Convert array to NumPy, regardless of source."""
    if hasattr(array, 'get'):
        return array.get()
    return np.asarray(array)


def from_numpy(array):
    """Convert NumPy array to current backend."""
    if CUDA_AVAILABLE:
        return _cupy.asarray(array)
    return array
