"""
GPU Profiling Script - Monitor GPU utilization during pathfinding

This script runs the pathfinding test while monitoring GPU metrics using nvidia-smi.
It helps verify that the GPU is being saturated (>50% utilization).

Usage:
    python profile_gpu_usage.py [--batch-size 32] [--duration 60]
"""

import subprocess
import threading
import time
import argparse
import logging
from collections import deque

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GPUMonitor:
    """Monitor GPU utilization using nvidia-smi"""

    def __init__(self, interval=0.5):
        self.interval = interval
        self.running = False
        self.utilization_samples = deque(maxlen=1000)
        self.memory_samples = deque(maxlen=1000)
        self.thread = None

    def start(self):
        """Start monitoring GPU"""
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        logger.info("GPU monitoring started")

    def stop(self):
        """Stop monitoring GPU"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        logger.info("GPU monitoring stopped")

    def _monitor_loop(self):
        """Monitor loop - polls nvidia-smi"""
        while self.running:
            try:
                # Query GPU utilization and memory
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
                     '--format=csv,noheader,nounits'],
                    capture_output=True,
                    text=True,
                    timeout=1.0
                )

                if result.returncode == 0:
                    output = result.stdout.strip()
                    if output:
                        parts = output.split(',')
                        if len(parts) >= 3:
                            util = float(parts[0].strip())
                            mem_used = float(parts[1].strip())
                            mem_total = float(parts[2].strip())

                            self.utilization_samples.append(util)
                            self.memory_samples.append(mem_used / mem_total * 100)

            except Exception as e:
                logger.debug(f"GPU monitoring error: {e}")

            time.sleep(self.interval)

    def get_stats(self):
        """Get GPU statistics"""
        if not self.utilization_samples:
            return None

        import numpy as np

        util_array = np.array(self.utilization_samples)
        mem_array = np.array(self.memory_samples)

        return {
            'avg_utilization': float(np.mean(util_array)),
            'max_utilization': float(np.max(util_array)),
            'min_utilization': float(np.min(util_array)),
            'avg_memory': float(np.mean(mem_array)),
            'max_memory': float(np.max(mem_array)),
            'samples': len(util_array)
        }


def run_pathfinding_test(batch_size=32, duration=60):
    """
    Run pathfinding test for specified duration.

    Args:
        batch_size: Number of ROIs to route in parallel
        duration: How long to run (seconds)
    """
    import sys
    import numpy as np

    try:
        import cupy as cp
        from orthoroute.algorithms.manhattan.pathfinder.cuda_dijkstra import CUDADijkstra
    except ImportError as e:
        logger.error(f"Cannot import CUDA modules: {e}")
        return False

    logger.info(f"Starting pathfinding test: batch_size={batch_size}, duration={duration}s")

    # Initialize solver
    cuda_solver = CUDADijkstra()

    # Create test data
    from test_parallel_gpu import create_test_roi_batch

    start_time = time.time()
    iterations = 0
    total_paths_found = 0
    total_rois_tested = 0

    try:
        while time.time() - start_time < duration:
            # Create batch
            roi_batch = create_test_roi_batch(batch_size, roi_size=100, density=0.15)

            # Run pathfinding
            paths = cuda_solver.find_paths_on_rois(roi_batch)

            # Count results
            found = sum(1 for p in paths if p)
            total_paths_found += found
            total_rois_tested += batch_size
            iterations += 1

            if iterations % 10 == 0:
                elapsed = time.time() - start_time
                throughput = total_rois_tested / elapsed
                logger.info(f"Iteration {iterations}: {total_paths_found}/{total_rois_tested} paths "
                           f"({throughput:.1f} ROIs/sec)")

        elapsed = time.time() - start_time
        throughput = total_rois_tested / elapsed

        logger.info(f"\nPathfinding test complete:")
        logger.info(f"  Iterations: {iterations}")
        logger.info(f"  Total ROIs: {total_rois_tested}")
        logger.info(f"  Paths found: {total_paths_found} ({total_paths_found/total_rois_tested*100:.1f}%)")
        logger.info(f"  Throughput: {throughput:.1f} ROIs/sec")
        logger.info(f"  Time per batch: {elapsed/iterations*1000:.1f} ms")

        return True

    except Exception as e:
        logger.error(f"Pathfinding test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Profile GPU usage during pathfinding')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Number of ROIs per batch (default: 32)')
    parser.add_argument('--duration', type=int, default=60,
                       help='Test duration in seconds (default: 60)')
    parser.add_argument('--no-monitor', action='store_true',
                       help='Disable GPU monitoring (just run test)')

    args = parser.parse_args()

    logger.info("="*80)
    logger.info("GPU PROFILING FOR PARALLEL PATHFINDING")
    logger.info("="*80)

    # Check if nvidia-smi is available
    if not args.no_monitor:
        try:
            result = subprocess.run(['nvidia-smi', '--version'],
                                   capture_output=True, timeout=1.0)
            if result.returncode != 0:
                logger.warning("nvidia-smi not available, disabling monitoring")
                args.no_monitor = True
        except Exception as e:
            logger.warning(f"nvidia-smi not available: {e}, disabling monitoring")
            args.no_monitor = True

    # Start GPU monitoring
    monitor = None
    if not args.no_monitor:
        monitor = GPUMonitor(interval=0.5)
        monitor.start()
        time.sleep(1.0)  # Let monitor start

    # Run pathfinding test
    success = run_pathfinding_test(
        batch_size=args.batch_size,
        duration=args.duration
    )

    # Stop monitoring
    if monitor:
        time.sleep(1.0)  # Get final samples
        monitor.stop()

        # Print statistics
        stats = monitor.get_stats()
        if stats:
            logger.info("\n" + "="*80)
            logger.info("GPU UTILIZATION STATISTICS")
            logger.info("="*80)
            logger.info(f"Samples collected: {stats['samples']}")
            logger.info(f"Average GPU utilization: {stats['avg_utilization']:.1f}%")
            logger.info(f"Peak GPU utilization: {stats['max_utilization']:.1f}%")
            logger.info(f"Min GPU utilization: {stats['min_utilization']:.1f}%")
            logger.info(f"Average memory usage: {stats['avg_memory']:.1f}%")
            logger.info(f"Peak memory usage: {stats['max_memory']:.1f}%")

            # Check if GPU is saturated
            if stats['avg_utilization'] > 50:
                logger.info(f"\n✓ SUCCESS: GPU is well utilized ({stats['avg_utilization']:.1f}% avg)")
            else:
                logger.warning(f"\n✗ WARNING: GPU utilization is low ({stats['avg_utilization']:.1f}% avg)")
                logger.warning("  This may indicate CPU bottlenecks or insufficient parallelism")

            if stats['max_utilization'] > 80:
                logger.info(f"✓ Peak utilization >80% - good saturation!")

    logger.info("\n" + "="*80)
    if success:
        logger.info("PROFILING COMPLETE")
    else:
        logger.error("PROFILING FAILED")
    logger.info("="*80)

    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
