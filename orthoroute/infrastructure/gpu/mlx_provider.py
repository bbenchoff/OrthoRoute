"""MLX GPU provider implementation for Apple Silicon."""
import logging
import platform
from typing import Optional, Dict, Any, Tuple
import numpy as np

from ...application.interfaces.gpu_provider import GPUProvider

logger = logging.getLogger(__name__)

# Check if we're on Apple Silicon
IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"

try:
    if IS_APPLE_SILICON:
        import mlx.core as mx
        MLX_AVAILABLE = True
    else:
        MLX_AVAILABLE = False
        mx = None
except ImportError:
    MLX_AVAILABLE = False
    mx = None


class MLXProvider(GPUProvider):
    """MLX/Metal implementation of GPU provider for Apple Silicon."""

    def __init__(self):
        """Initialize MLX provider."""
        self._mlx = None
        self._device_info = {}
        self._initialized = False
        self._allocated_arrays = []

    def is_available(self) -> bool:
        """Check if MLX is available."""
        if not IS_APPLE_SILICON:
            logger.warning("MLX requires Apple Silicon - not available on this platform")
            return False

        if not MLX_AVAILABLE:
            logger.warning("MLX not installed - Apple GPU acceleration unavailable")
            return False

        try:
            # Test basic MLX functionality
            test_array = mx.array([1, 2, 3])
            _ = mx.sum(test_array)
            # Synchronize to ensure GPU works
            mx.eval(_)
            logger.info("MLX GPU (Apple Silicon) detected and working")
            return True
        except Exception as e:
            logger.warning(f"MLX error: {str(e)} - Apple GPU acceleration unavailable")
            return False

    def initialize(self) -> bool:
        """Initialize MLX resources."""
        if self._initialized:
            return True

        if not MLX_AVAILABLE:
            logger.error("MLX not available")
            return False

        try:
            self._mlx = mx

            # Get device information
            # MLX doesn't expose detailed GPU info, but we can get basic info
            import subprocess
            try:
                result = subprocess.run(
                    ["system_profiler", "SPDisplaysDataType"],
                    capture_output=True, text=True, timeout=5
                )
                gpu_info = result.stdout if result.returncode == 0 else "Apple GPU"
            except Exception:
                gpu_info = "Apple GPU"

            # Get memory info from system
            try:
                import psutil
                total_mem = psutil.virtual_memory().total
                free_mem = psutil.virtual_memory().available
            except ImportError:
                total_mem = 0
                free_mem = 0

            self._device_info = {
                'name': 'Apple Silicon GPU (MLX)',
                'compute_capability': 'Metal',
                'total_memory': total_mem,
                'free_memory': free_mem,
                'memory_limit': int(free_mem * 0.8),
                'device_id': 'mlx'
            }

            # Test basic operations
            test_array = mx.ones((100, 100), dtype=mx.float32)
            result = mx.sum(test_array)
            mx.eval(result)

            self._initialized = True
            logger.info(f"MLX initialized: Apple Silicon GPU")
            logger.info(f"Memory: {total_mem / 1024**3:.1f}GB total, {free_mem / 1024**3:.1f}GB free")

            return True

        except Exception as e:
            logger.error(f"Failed to initialize MLX: {e}")
            return False

    def cleanup(self) -> None:
        """Cleanup MLX resources."""
        # Clear array references to help garbage collection
        self._allocated_arrays.clear()
        self._initialized = False
        logger.debug("MLX provider cleaned up")

    def get_device_info(self) -> Dict[str, Any]:
        """Get MLX device information."""
        return self._device_info.copy()

    def get_memory_info(self) -> Dict[str, Any]:
        """Get memory usage information."""
        if not self._initialized:
            return {
                'total_memory': 0,
                'free_memory': 0,
                'used_memory': 0,
                'memory_pool_used': 0,
                'memory_pool_total': 0
            }

        try:
            import psutil
            vm = psutil.virtual_memory()

            # Estimate memory used by MLX arrays
            array_memory = sum(
                arr.nbytes if hasattr(arr, 'nbytes') else 0
                for arr in self._allocated_arrays
            )

            return {
                'total_memory': vm.total,
                'free_memory': vm.available,
                'used_memory': vm.used,
                'memory_pool_used': array_memory,
                'memory_pool_total': array_memory
            }

        except Exception as e:
            logger.error(f"Error getting memory info: {e}")
            return {
                'total_memory': 0,
                'free_memory': 0,
                'used_memory': 0,
                'memory_pool_used': 0,
                'memory_pool_total': 0
            }

    def create_array(self, shape: Tuple[int, ...], dtype=None, fill_value=None) -> Any:
        """Create array using MLX."""
        if not self._initialized or not self._mlx:
            raise RuntimeError("MLX not initialized")

        # Convert numpy dtype to MLX dtype
        mlx_dtype = self._numpy_to_mlx_dtype(dtype)

        try:
            if fill_value is None:
                # MLX doesn't have empty, use zeros
                array = mx.zeros(shape, dtype=mlx_dtype)
            elif fill_value == 0:
                array = mx.zeros(shape, dtype=mlx_dtype)
            elif fill_value == 1:
                array = mx.ones(shape, dtype=mlx_dtype)
            else:
                array = mx.full(shape, fill_value, dtype=mlx_dtype)

            # Track allocated arrays
            self._allocated_arrays.append(array)

            return array

        except Exception as e:
            logger.error(f"Error creating MLX array: {e}")
            raise

    def copy_array(self, array: Any) -> Any:
        """Create copy of array."""
        if not self._initialized or not self._mlx:
            raise RuntimeError("MLX not initialized")

        try:
            # MLX arrays are immutable, so we can use array directly
            # But for compatibility, we create a copy by adding 0
            if hasattr(array, '__class__') and 'mlx' in str(type(array)):
                copied = mx.array(array)
            else:
                copied = mx.array(array)
            self._allocated_arrays.append(copied)
            return copied

        except Exception as e:
            logger.error(f"Error copying MLX array: {e}")
            raise

    def to_cpu(self, array: Any) -> np.ndarray:
        """Convert MLX array to CPU numpy array."""
        if not self._initialized or not self._mlx:
            return array if isinstance(array, np.ndarray) else np.array(array)

        try:
            if hasattr(array, '__class__') and 'mlx' in str(type(array)):
                # Ensure computation is complete before converting
                mx.eval(array)
                return np.array(array)
            else:
                return array if isinstance(array, np.ndarray) else np.array(array)
        except Exception as e:
            logger.error(f"Error converting array to CPU: {e}")
            return array if isinstance(array, np.ndarray) else np.array(array)

    def to_gpu(self, array: np.ndarray) -> Any:
        """Convert CPU array to MLX array."""
        if not self._initialized or not self._mlx:
            raise RuntimeError("MLX not initialized")

        try:
            return mx.array(array)
        except Exception as e:
            logger.error(f"Error converting array to MLX: {e}")
            raise

    def synchronize(self) -> None:
        """Synchronize MLX operations."""
        if not self._initialized or not self._mlx:
            return

        try:
            # MLX uses lazy evaluation, so we need to sync any pending operations
            # This is typically done by calling mx.eval() on arrays
            pass  # No global sync needed in MLX, operations are synced on access
        except Exception as e:
            logger.error(f"Error synchronizing MLX: {e}")

    def _numpy_to_mlx_dtype(self, dtype):
        """Convert numpy dtype to MLX dtype."""
        if dtype is None:
            return mx.float32

        # Handle numpy dtypes
        dtype_map = {
            np.float32: mx.float32,
            np.float64: mx.float32,  # MLX prefers float32
            np.int32: mx.int32,
            np.int64: mx.int64,
            np.int16: mx.int16,
            np.int8: mx.int8,
            np.uint8: mx.uint8,
            np.uint16: mx.uint16,
            np.uint32: mx.uint32,
            np.uint64: mx.uint64,
            np.bool_: mx.bool_,
        }

        # Handle dtype instances
        if hasattr(dtype, 'type'):
            dtype = dtype.type

        return dtype_map.get(dtype, mx.float32)

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
