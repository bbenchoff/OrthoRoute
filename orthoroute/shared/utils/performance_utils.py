"""Performance monitoring utilities."""
import time
import logging
import psutil
import functools
from contextlib import contextmanager
from typing import Generator, Dict, Any, Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    execution_time: float = 0.0
    memory_start_mb: float = 0.0
    memory_end_mb: float = 0.0
    memory_peak_mb: float = 0.0
    cpu_percent: float = 0.0
    additional_metrics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def memory_delta_mb(self) -> float:
        """Memory change during execution."""
        return self.memory_end_mb - self.memory_start_mb
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'execution_time': self.execution_time,
            'memory_start_mb': self.memory_start_mb,
            'memory_end_mb': self.memory_end_mb,
            'memory_peak_mb': self.memory_peak_mb,
            'memory_delta_mb': self.memory_delta_mb,
            'cpu_percent': self.cpu_percent,
            **self.additional_metrics
        }


@contextmanager
def timing_context(name: str = "operation", log_result: bool = True) -> Generator[PerformanceMetrics, None, None]:
    """Context manager for timing operations.
    
    Args:
        name: Name of the operation being timed
        log_result: Whether to log the timing result
        
    Yields:
        PerformanceMetrics object that gets populated during execution
    """
    metrics = PerformanceMetrics()
    
    # Record starting metrics
    start_time = time.perf_counter()
    process = psutil.Process()
    
    try:
        memory_info = process.memory_info()
        metrics.memory_start_mb = memory_info.rss / 1024 / 1024
        metrics.memory_peak_mb = metrics.memory_start_mb
    except Exception:
        # If psutil fails, just continue without memory tracking
        pass
    
    try:
        yield metrics
    finally:
        # Record ending metrics
        end_time = time.perf_counter()
        metrics.execution_time = end_time - start_time
        
        try:
            memory_info = process.memory_info()
            metrics.memory_end_mb = memory_info.rss / 1024 / 1024
            metrics.cpu_percent = process.cpu_percent()
            
            # Update peak memory if current is higher
            if metrics.memory_end_mb > metrics.memory_peak_mb:
                metrics.memory_peak_mb = metrics.memory_end_mb
                
        except Exception:
            # If psutil fails, just continue
            pass
        
        if log_result:
            logger.info(f"{name} completed in {metrics.execution_time:.3f}s")
            if metrics.memory_start_mb > 0:
                logger.debug(f"{name} memory: {metrics.memory_delta_mb:+.1f}MB "
                           f"(peak: {metrics.memory_peak_mb:.1f}MB)")


def performance_monitor(func: Callable = None, *, name: str = None, log_result: bool = True):
    """Decorator for monitoring function performance.
    
    Args:
        func: Function to decorate
        name: Optional name for the operation
        log_result: Whether to log performance results
        
    Returns:
        Decorated function that returns (result, metrics) tuple
    """
    def decorator(f):
        operation_name = name or f.__name__
        
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            with timing_context(operation_name, log_result) as metrics:
                result = f(*args, **kwargs)
            return result, metrics
        
        return wrapper
    
    if func is None:
        # Called with arguments
        return decorator
    else:
        # Called without arguments
        return decorator(func)


@contextmanager
def memory_profiler(name: str = "operation", log_result: bool = True) -> Generator[PerformanceMetrics, None, None]:
    """Context manager for detailed memory profiling.
    
    Args:
        name: Name of the operation being profiled
        log_result: Whether to log profiling results
        
    Yields:
        PerformanceMetrics object with detailed memory info
    """
    metrics = PerformanceMetrics()
    
    # Get process handle
    process = psutil.Process()
    
    try:
        # Record initial memory state
        memory_info = process.memory_info()
        metrics.memory_start_mb = memory_info.rss / 1024 / 1024
        metrics.memory_peak_mb = metrics.memory_start_mb
        
        # Additional memory details
        try:
            memory_percent = process.memory_percent()
            metrics.additional_metrics['memory_percent'] = memory_percent
        except Exception:
            pass
        
        start_time = time.perf_counter()
        
        yield metrics
        
    finally:
        # Record final memory state
        end_time = time.perf_counter()
        metrics.execution_time = end_time - start_time
        
        try:
            memory_info = process.memory_info()
            metrics.memory_end_mb = memory_info.rss / 1024 / 1024
            metrics.cpu_percent = process.cpu_percent()
            
            # Get memory percent
            memory_percent = process.memory_percent()
            metrics.additional_metrics['memory_percent_end'] = memory_percent
            
            # Check if memory increased significantly
            if metrics.memory_delta_mb > 10:  # More than 10MB increase
                logger.warning(f"{name} increased memory by {metrics.memory_delta_mb:.1f}MB")
            
        except Exception:
            pass
        
        if log_result:
            logger.info(f"{name} memory profile: "
                       f"start={metrics.memory_start_mb:.1f}MB "
                       f"end={metrics.memory_end_mb:.1f}MB "
                       f"delta={metrics.memory_delta_mb:+.1f}MB "
                       f"time={metrics.execution_time:.3f}s")


class PerformanceTracker:
    """Class for tracking performance across multiple operations."""
    
    def __init__(self):
        """Initialize performance tracker."""
        self.metrics: Dict[str, PerformanceMetrics] = {}
        self.operation_counts: Dict[str, int] = {}
    
    @contextmanager
    def track_operation(self, name: str, log_result: bool = False):
        """Track a named operation.
        
        Args:
            name: Operation name
            log_result: Whether to log individual results
        """
        with timing_context(name, log_result) as metrics:
            yield metrics
        
        # Store metrics
        if name not in self.metrics:
            self.metrics[name] = metrics
            self.operation_counts[name] = 1
        else:
            # Accumulate metrics for repeated operations
            prev_metrics = self.metrics[name]
            count = self.operation_counts[name] + 1
            
            # Calculate running averages
            new_metrics = PerformanceMetrics(
                execution_time=(prev_metrics.execution_time * (count - 1) + metrics.execution_time) / count,
                memory_delta_mb=(prev_metrics.memory_delta_mb * (count - 1) + metrics.memory_delta_mb) / count,
                cpu_percent=(prev_metrics.cpu_percent * (count - 1) + metrics.cpu_percent) / count
            )
            
            # Keep peak values
            new_metrics.memory_peak_mb = max(prev_metrics.memory_peak_mb, metrics.memory_peak_mb)
            
            self.metrics[name] = new_metrics
            self.operation_counts[name] = count
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        summary = {}
        
        for name, metrics in self.metrics.items():
            count = self.operation_counts[name]
            summary[name] = {
                'count': count,
                'avg_execution_time': metrics.execution_time,
                'total_execution_time': metrics.execution_time * count,
                'avg_memory_delta_mb': metrics.memory_delta_mb,
                'peak_memory_mb': metrics.memory_peak_mb,
                'avg_cpu_percent': metrics.cpu_percent
            }
        
        return summary
    
    def log_summary(self):
        """Log performance summary."""
        summary = self.get_summary()
        
        logger.info("Performance Summary:")
        for name, stats in summary.items():
            logger.info(f"  {name}: "
                       f"count={stats['count']} "
                       f"avg_time={stats['avg_execution_time']:.3f}s "
                       f"total_time={stats['total_execution_time']:.3f}s "
                       f"avg_memory={stats['avg_memory_delta_mb']:+.1f}MB")
    
    def reset(self):
        """Reset all tracked metrics."""
        self.metrics.clear()
        self.operation_counts.clear()