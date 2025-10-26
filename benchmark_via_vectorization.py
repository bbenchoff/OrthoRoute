#!/usr/bin/env python3
"""
Benchmark vectorized vs sequential via pooling penalty application
"""

import sys
import numpy as np
import time
from pathlib import Path

# Add the package directory to Python path
package_dir = Path(__file__).parent
if str(package_dir) not in sys.path:
    sys.path.insert(0, str(package_dir))


def benchmark_penalty_computation(num_via_edges_list, grid_size=100, num_layers=18):
    """Benchmark vectorized vs sequential implementations with various edge counts"""

    print("\n" + "=" * 80)
    print("VIA POOLING PENALTY PERFORMANCE BENCHMARK")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Grid size: {grid_size}x{grid_size}")
    print(f"  Number of layers: {num_layers}")
    print(f"  Segment layers: {num_layers - 2}")
    print()

    results = []

    for num_via_edges in num_via_edges_list:
        print("-" * 80)
        print(f"\nTesting with {num_via_edges:,} via edges:")

        # Setup test data
        np.random.seed(42)

        via_edge_indices = np.arange(num_via_edges, dtype=np.int32)
        via_xy_coords = np.random.randint(0, grid_size, size=(num_via_edges, 2), dtype=np.int32)
        z_lo = np.random.randint(1, num_layers - 1, size=num_via_edges, dtype=np.int32)
        z_hi = np.random.randint(2, num_layers, size=num_via_edges, dtype=np.int32)
        z_hi = np.maximum(z_lo + 1, z_hi)

        # Create mock pressure arrays
        via_col_pres = np.random.rand(grid_size, grid_size).astype(np.float32) * 2.0
        via_seg_pres = np.random.rand(grid_size, grid_size, num_layers - 2).astype(np.float32)
        via_seg_prefix = np.cumsum(via_seg_pres, axis=2)

        col_weight = 1.0
        seg_weight = 1.0
        pres_fac = 2.0

        # Mock cost arrays
        total_cost_vec = np.random.rand(num_via_edges * 2).astype(np.float32)
        total_cost_seq = total_cost_vec.copy()

        # =====================================================================
        # VECTORIZED IMPLEMENTATION
        # =====================================================================
        t0_vec = time.perf_counter()

        penalties_vec = np.zeros(num_via_edges, dtype=np.float32)

        # Column penalties
        col_penalties = via_col_pres[via_xy_coords[:, 0], via_xy_coords[:, 1]]
        penalties_vec += col_weight * col_penalties

        # Segment penalties
        hi_idx = z_hi - 2
        lo_idx = z_lo - 2
        valid_mask = z_hi > z_lo
        hi_valid = (hi_idx >= 0) & (hi_idx < (num_layers - 2))
        lo_valid = (lo_idx >= 0) & (lo_idx < (num_layers - 2))

        pref_hi = np.zeros(num_via_edges, dtype=np.float32)
        pref_lo = np.zeros(num_via_edges, dtype=np.float32)

        if np.any(hi_valid):
            pref_hi[hi_valid] = via_seg_prefix[
                via_xy_coords[hi_valid, 0],
                via_xy_coords[hi_valid, 1],
                hi_idx[hi_valid]
            ]

        if np.any(lo_valid):
            pref_lo[lo_valid] = via_seg_prefix[
                via_xy_coords[lo_valid, 0],
                via_xy_coords[lo_valid, 1],
                lo_idx[lo_valid]
            ]

        seg_penalties = (pref_hi - pref_lo) * valid_mask
        penalties_vec += seg_weight * seg_penalties

        # Apply penalties
        penalty_mask = penalties_vec > 0
        total_cost_vec[via_edge_indices[penalty_mask]] += pres_fac * penalties_vec[penalty_mask]
        penalties_applied_vec = np.sum(penalty_mask)

        t1_vec = time.perf_counter()
        time_vec = t1_vec - t0_vec

        # =====================================================================
        # SEQUENTIAL IMPLEMENTATION
        # =====================================================================
        t0_seq = time.perf_counter()

        penalties_applied_seq = 0

        for i in range(num_via_edges):
            ei = via_edge_indices[i]
            x, y = via_xy_coords[i]
            penalty = 0.0

            # Column penalty
            penalty += col_weight * via_col_pres[x, y]

            # Segment penalty
            zlo, zhi = z_lo[i], z_hi[i]
            if zhi > zlo:
                hi_idx_seq = zhi - 2
                lo_idx_seq = zlo - 2
                pref_hi_seq = via_seg_prefix[x, y, hi_idx_seq] if 0 <= hi_idx_seq < (num_layers - 2) else 0.0
                pref_lo_seq = via_seg_prefix[x, y, lo_idx_seq] if 0 <= lo_idx_seq < (num_layers - 2) else 0.0
                seg_sum = pref_hi_seq - pref_lo_seq
                penalty += seg_weight * seg_sum

            if penalty > 0:
                total_cost_seq[ei] += pres_fac * penalty
                penalties_applied_seq += 1

        t1_seq = time.perf_counter()
        time_seq = t1_seq - t0_seq

        # =====================================================================
        # RESULTS
        # =====================================================================
        speedup = time_seq / time_vec if time_vec > 0 else 0

        print(f"\n  Vectorized:  {time_vec:.4f}s ({penalties_applied_vec} penalties applied)")
        print(f"  Sequential:  {time_seq:.4f}s ({penalties_applied_seq} penalties applied)")
        print(f"  Speedup:     {speedup:.1f}x")
        print(f"  Throughput:  {num_via_edges / time_vec:,.0f} edges/sec (vectorized)")
        print(f"               {num_via_edges / time_seq:,.0f} edges/sec (sequential)")

        # Verify correctness
        cost_diff = np.abs(total_cost_vec - total_cost_seq)
        max_cost_diff = np.max(cost_diff)
        if max_cost_diff < 1e-5:
            print(f"  Correctness: PASS (max diff: {max_cost_diff:.2e})")
        else:
            print(f"  Correctness: FAIL (max diff: {max_cost_diff:.2e})")

        results.append({
            'num_edges': num_via_edges,
            'time_vec': time_vec,
            'time_seq': time_seq,
            'speedup': speedup,
        })

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n{'Edges':>12} | {'Vectorized':>12} | {'Sequential':>12} | {'Speedup':>10}")
    print("-" * 80)
    for r in results:
        print(f"{r['num_edges']:>12,} | {r['time_vec']:>10.4f}s | {r['time_seq']:>10.4f}s | {r['speedup']:>9.1f}x")

    avg_speedup = np.mean([r['speedup'] for r in results])
    print("-" * 80)
    print(f"\nAverage speedup: {avg_speedup:.1f}x")

    # Estimate time for 50M edges
    if len(results) >= 2:
        # Use last two results to estimate scaling
        last_result = results[-1]
        time_per_edge_vec = last_result['time_vec'] / last_result['num_edges']
        time_per_edge_seq = last_result['time_seq'] / last_result['num_edges']

        edges_50m = 50_000_000
        estimated_time_vec = time_per_edge_vec * edges_50m
        estimated_time_seq = time_per_edge_seq * edges_50m

        print(f"\nEstimated time for 50M edges:")
        print(f"  Vectorized:  {estimated_time_vec:.2f}s ({estimated_time_vec/60:.1f} min)")
        print(f"  Sequential:  {estimated_time_seq:.2f}s ({estimated_time_seq/60:.1f} min)")
        print(f"  Time saved:  {estimated_time_seq - estimated_time_vec:.2f}s ({(estimated_time_seq - estimated_time_vec)/60:.1f} min)")

    print("\n" + "=" * 80)


def main():
    """Run benchmark with increasing edge counts"""

    # Test with progressively larger datasets
    edge_counts = [
        1_000,        # 1K edges - warmup
        10_000,       # 10K edges
        100_000,      # 100K edges
        1_000_000,    # 1M edges
        5_000_000,    # 5M edges (realistic for large boards)
    ]

    benchmark_penalty_computation(edge_counts)


if __name__ == "__main__":
    main()
