"""
Performance benchmark for CUDA Near-Far algorithm.
Compares CPU heapq Dijkstra vs GPU Near-Far on various ROI sizes.
"""

import time
import numpy as np

try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    print("CuPy not available - cannot benchmark CUDA Near-Far")
    CUDA_AVAILABLE = False
    exit(1)

from orthoroute.algorithms.manhattan.pathfinder.cuda_dijkstra import CUDADijkstra


def generate_grid_graph(grid_size):
    """
    Generate 4-connected grid graph (PCB-like).

    Args:
        grid_size: Total nodes (will be sqrt(grid_size) × sqrt(grid_size) grid)

    Returns:
        indptr, indices, weights (CSR format)
    """
    import math

    # Make square grid
    side = int(math.sqrt(grid_size))
    n = side * side

    edges = []

    # Build 4-connected grid
    for row in range(side):
        for col in range(side):
            node = row * side + col

            # Right neighbor
            if col < side - 1:
                neighbor = node + 1
                edges.append((node, neighbor, 0.4))
                edges.append((neighbor, node, 0.4))  # Bidirectional

            # Down neighbor
            if row < side - 1:
                neighbor = node + side
                edges.append((node, neighbor, 0.4))
                edges.append((neighbor, node, 0.4))  # Bidirectional

    # Sort edges by source node
    edges.sort(key=lambda x: (x[0], x[1]))

    # Build CSR
    indptr = np.zeros(n + 1, dtype=np.int32)
    indices = np.zeros(len(edges), dtype=np.int32)
    weights = np.zeros(len(edges), dtype=np.float32)

    for i, (u, v, w) in enumerate(edges):
        indptr[u + 1] += 1
        indices[i] = v
        weights[i] = w

    indptr = np.cumsum(indptr)

    return indptr, indices, weights, n


def cpu_dijkstra(indptr, indices, weights, src, dst, size):
    """Reference CPU Dijkstra implementation"""
    import heapq

    dist = [float('inf')] * size
    parent = [-1] * size
    dist[src] = 0.0

    heap = [(0.0, src)]
    visited = set()

    while heap:
        d, u = heapq.heappop(heap)

        if u in visited:
            continue

        visited.add(u)

        if u == dst:
            break

        for i in range(indptr[u], indptr[u + 1]):
            v = indices[i]
            cost = weights[i]

            if v not in visited:
                new_dist = d + cost
                if new_dist < dist[v]:
                    dist[v] = new_dist
                    parent[v] = u
                    heapq.heappush(heap, (new_dist, v))

    # Reconstruct path
    if dist[dst] == float('inf'):
        return None

    path = []
    curr = dst
    while curr != -1:
        path.append(curr)
        curr = parent[curr]
    path.reverse()
    return path


def benchmark_single_roi(target_size):
    """Benchmark CPU vs GPU on single ROI of given size"""

    print(f"\n{'='*70}")
    print(f"Benchmarking ROI Size: {target_size} nodes")
    print(f"{'='*70}")

    # Generate graph
    indptr, indices, weights, actual_size = generate_grid_graph(target_size)
    print(f"Generated grid: {actual_size} nodes, {len(indices)} edges")

    src = 0
    dst = actual_size - 1

    # Warmup
    _ = cpu_dijkstra(indptr, indices, weights, src, dst, actual_size)

    # CPU benchmark (5 runs)
    print("\nCPU (heapq Dijkstra):")
    cpu_times = []
    for run in range(5):
        start = time.perf_counter()
        cpu_path = cpu_dijkstra(indptr, indices, weights, src, dst, actual_size)
        elapsed = (time.perf_counter() - start) * 1000
        cpu_times.append(elapsed)
        print(f"  Run {run+1}: {elapsed:.2f} ms")

    cpu_avg = np.mean(cpu_times)
    cpu_std = np.std(cpu_times)
    print(f"  Average: {cpu_avg:.2f} ± {cpu_std:.2f} ms")
    print(f"  Path length: {len(cpu_path)}")

    # GPU benchmark (5 runs)
    print("\nGPU (Near-Far):")
    solver = CUDADijkstra()

    # Warmup
    _ = solver.find_paths_on_rois([
        (src, dst, cp.asarray(indptr), cp.asarray(indices),
         cp.asarray(weights), actual_size)
    ])

    gpu_times = []
    for run in range(5):
        start = time.perf_counter()
        gpu_paths = solver.find_paths_on_rois([
            (src, dst, cp.asarray(indptr), cp.asarray(indices),
             cp.asarray(weights), actual_size)
        ])
        elapsed = (time.perf_counter() - start) * 1000
        gpu_times.append(elapsed)
        print(f"  Run {run+1}: {elapsed:.2f} ms")

    gpu_avg = np.mean(gpu_times)
    gpu_std = np.std(gpu_times)
    print(f"  Average: {gpu_avg:.2f} ± {gpu_std:.2f} ms")
    print(f"  Path length: {len(gpu_paths[0])}")

    # Speedup
    speedup = cpu_avg / gpu_avg
    print(f"\n{'='*70}")
    print(f"SPEEDUP: {speedup:.1f}×")
    print(f"{'='*70}")

    # Verify correctness
    if cpu_path == gpu_paths[0]:
        print("✓ Correctness: GPU == CPU (paths identical)")
    else:
        print("✗ WARNING: GPU != CPU (paths differ!)")
        print(f"  CPU: {cpu_path[:10]}...")
        print(f"  GPU: {gpu_paths[0][:10]}...")

    return {
        'size': actual_size,
        'edges': len(indices),
        'cpu_avg': cpu_avg,
        'cpu_std': cpu_std,
        'gpu_avg': gpu_avg,
        'gpu_std': gpu_std,
        'speedup': speedup,
        'path_length': len(cpu_path)
    }


def benchmark_batch_sizes():
    """Benchmark effect of batch size (K ROIs)"""

    print(f"\n{'='*70}")
    print("Batch Size Benchmark")
    print(f"{'='*70}")

    base_size = 10000
    batch_sizes = [1, 2, 4, 8, 16]

    solver = CUDADijkstra()

    results = []

    for K in batch_sizes:
        print(f"\nBatch size K = {K}")

        # Generate K ROIs
        roi_batch = []
        for i in range(K):
            indptr, indices, weights, size = generate_grid_graph(base_size)
            roi_batch.append((0, size-1, cp.asarray(indptr),
                            cp.asarray(indices), cp.asarray(weights), size))

        # Benchmark
        times = []
        for run in range(3):
            start = time.perf_counter()
            paths = solver.find_paths_on_rois(roi_batch)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        avg_time = np.mean(times)
        per_roi = avg_time / K

        print(f"  Total time: {avg_time:.2f} ms")
        print(f"  Per ROI: {per_roi:.2f} ms")
        print(f"  Throughput: {K / avg_time * 1000:.1f} ROIs/sec")

        results.append({
            'batch_size': K,
            'total_time': avg_time,
            'per_roi': per_roi,
            'throughput': K / avg_time * 1000
        })

    print(f"\n{'='*70}")
    print("Batch Size Summary")
    print(f"{'='*70}")
    print(f"{'K':>4} | {'Total (ms)':>12} | {'Per ROI (ms)':>14} | {'Throughput (ROIs/s)':>20}")
    print(f"{'-'*70}")

    for r in results:
        print(f"{r['batch_size']:>4} | {r['total_time']:>12.2f} | "
              f"{r['per_roi']:>14.2f} | {r['throughput']:>20.1f}")


def main():
    print("="*70)
    print("CUDA Near-Far Performance Benchmark")
    print("="*70)

    if not CUDA_AVAILABLE:
        print("CUDA not available - exiting")
        return

    # Single ROI benchmarks
    sizes = [1000, 5000, 10000, 25000, 50000]

    all_results = []

    for size in sizes:
        result = benchmark_single_roi(size)
        all_results.append(result)

    # Summary table
    print(f"\n{'='*70}")
    print("Performance Summary")
    print(f"{'='*70}")
    print(f"{'Size':>8} | {'Edges':>8} | {'CPU (ms)':>10} | "
          f"{'GPU (ms)':>10} | {'Speedup':>8}")
    print(f"{'-'*70}")

    for r in all_results:
        print(f"{r['size']:>8} | {r['edges']:>8} | "
              f"{r['cpu_avg']:>10.1f} | {r['gpu_avg']:>10.1f} | "
              f"{r['speedup']:>8.1f}×")

    # Batch size benchmark
    benchmark_batch_sizes()

    print(f"\n{'='*70}")
    print("Benchmark Complete")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
