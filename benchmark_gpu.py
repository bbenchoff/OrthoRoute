#!/usr/bin/env python3
"""
GPU Router Performance Benchmark
Validates weekend plan goals: 10-127× speedup target
"""
import time
import sys
import os
import subprocess
import logging
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('benchmark_gpu.log'),
        logging.StreamHandler()
    ]
)

def print_header(title):
    """Print formatted section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

def run_test_with_timing(env_vars, phase_name):
    """Run test-manhattan with specified environment variables and measure time."""
    print(f"\n[{phase_name}] Starting test...")

    # Build environment
    test_env = os.environ.copy()
    test_env.update(env_vars)

    # Log environment config
    gpu_mode = env_vars.get('GPU_PERSISTENT_ROUTER', '0')
    print(f"[{phase_name}] Environment: GPU_PERSISTENT_ROUTER={gpu_mode}")

    # Run test and capture output
    start_time = time.time()

    cmd = [sys.executable, 'main.py', '--test-manhattan', '--min-run-sec', '0']

    try:
        result = subprocess.run(
            cmd,
            env=test_env,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )

        elapsed = time.time() - start_time

        # Parse output for metrics
        output = result.stdout + result.stderr

        # Extract key metrics
        metrics = {
            'elapsed_sec': elapsed,
            'success': result.returncode == 0,
            'nets_routed': 0,
            'total_nets': 0,
            'throughput': 0.0
        }

        # Parse output for stats
        for line in output.split('\n'):
            if 'nets routed' in line.lower() or 'routed=' in line.lower():
                # Try to extract routed/total
                try:
                    if 'routed=' in line:
                        # Format: "routed=464/464"
                        parts = line.split('routed=')[1].split()[0]
                        routed, total = parts.split('/')
                        metrics['nets_routed'] = int(routed)
                        metrics['total_nets'] = int(total)
                except:
                    pass

            if 'nets/sec' in line.lower():
                try:
                    # Extract throughput
                    parts = line.split('nets/sec')
                    val_str = parts[0].split()[-1]
                    metrics['throughput'] = float(val_str)
                except:
                    pass

        # Log results
        print(f"[{phase_name}] Completed in {elapsed:.1f} seconds")
        print(f"[{phase_name}] Success: {metrics['success']}")
        print(f"[{phase_name}] Nets routed: {metrics['nets_routed']}/{metrics['total_nets']}")
        print(f"[{phase_name}] Throughput: {metrics['throughput']:.2f} nets/sec")

        # Save full output
        output_file = f"benchmark_{phase_name.lower().replace(' ', '_')}.txt"
        with open(output_file, 'w') as f:
            f.write(output)
        print(f"[{phase_name}] Full output saved to: {output_file}")

        return metrics

    except subprocess.TimeoutExpired:
        print(f"[{phase_name}] ERROR: Test timed out after 1 hour")
        return None
    except Exception as e:
        print(f"[{phase_name}] ERROR: {e}")
        return None

def analyze_gpu_utilization(phase_name):
    """Attempt to get GPU utilization statistics."""
    print(f"\n[{phase_name}] Checking GPU utilization...")

    try:
        # Try nvidia-smi
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            output = result.stdout.strip()
            util, mem_used, mem_total = output.split(',')
            print(f"[{phase_name}] GPU Utilization: {util.strip()}%")
            print(f"[{phase_name}] GPU Memory: {mem_used.strip()}/{mem_total.strip()} MB")
            return {
                'utilization_pct': float(util.strip()),
                'memory_used_mb': float(mem_used.strip()),
                'memory_total_mb': float(mem_total.strip())
            }
    except Exception as e:
        print(f"[{phase_name}] Could not get GPU stats: {e}")

    return None

def generate_report(baseline_metrics, gpu_metrics):
    """Generate comprehensive performance report."""
    print_header("PERFORMANCE REPORT")

    # Check if we have valid metrics
    if not baseline_metrics or not gpu_metrics:
        print("ERROR: Could not complete benchmark - missing metrics")
        return

    if not baseline_metrics.get('success') or not gpu_metrics.get('success'):
        print("ERROR: One or both tests failed")
        print(f"  Baseline success: {baseline_metrics.get('success')}")
        print(f"  GPU success: {gpu_metrics.get('success')}")
        return

    # Calculate speedup
    baseline_time = baseline_metrics['elapsed_sec']
    gpu_time = gpu_metrics['elapsed_sec']

    if gpu_time > 0:
        speedup = baseline_time / gpu_time
    else:
        speedup = float('inf')

    # Print comparison table
    print("┌─────────────────────────────────────────────────────────────────┐")
    print("│                    BASELINE VS GPU COMPARISON                   │")
    print("├─────────────────────────────────────────────────────────────────┤")
    print(f"│ Metric              │ Baseline (CPU)  │ GPU Persistent   │")
    print("├─────────────────────────────────────────────────────────────────┤")
    print(f"│ Execution Time      │ {baseline_time:8.1f} sec    │ {gpu_time:8.1f} sec   │")
    print(f"│ Nets Routed         │ {baseline_metrics['nets_routed']:6d}/{baseline_metrics['total_nets']:6d}   │ {gpu_metrics['nets_routed']:6d}/{gpu_metrics['total_nets']:6d}  │")
    print(f"│ Throughput          │ {baseline_metrics['throughput']:8.2f} n/s   │ {gpu_metrics['throughput']:8.2f} n/s  │")
    print(f"│ Speedup             │        -        │ {speedup:8.1f}×      │")
    print("└─────────────────────────────────────────────────────────────────┘")

    # Weekend plan goal comparison
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│                  WEEKEND PLAN GOALS VALIDATION                  │")
    print("├─────────────────────────────────────────────────────────────────┤")
    print(f"│ Target Speedup (min) │ 10×             │ {'✓ PASS' if speedup >= 10 else '✗ FAIL'}         │")
    print(f"│ Target Speedup (max) │ 127×            │ {speedup/127*100:6.1f}% achieved│")
    print(f"│ Actual Speedup       │ {speedup:.1f}×            │                │")
    print("└─────────────────────────────────────────────────────────────────┘")

    # Goal status
    print("\n" + "="*80)
    if speedup >= 10:
        print("  ✓ SUCCESS: Minimum 10× speedup goal achieved!")
        print(f"  Achievement: {speedup:.1f}× speedup ({speedup/127*100:.1f}% of 127× theoretical max)")
    else:
        print("  ✗ FAILED: Did not achieve minimum 10× speedup goal")
        print(f"  Achievement: Only {speedup:.1f}× speedup (need 10×)")
    print("="*80)

    # Bottleneck analysis if < 10×
    if speedup < 10:
        print("\n┌─────────────────────────────────────────────────────────────────┐")
        print("│                      BOTTLENECK ANALYSIS                        │")
        print("├─────────────────────────────────────────────────────────────────┤")
        print("│ Possible causes:                                                │")
        print("│ • ROI extraction overhead (should be ~0 with persistent mode)   │")
        print("│ • Kernel launch overhead (check for multiple launches)          │")
        print("│ • Host-device transfer time (should be minimal)                 │")
        print("│ • Backtrace time (check if device-side backtrace is active)     │")
        print("│ • GPU not fully utilized (check nvidia-smi during run)          │")
        print("└─────────────────────────────────────────────────────────────────┘")
        print("\nRecommendations:")
        print("  1. Check logs for 'ROI extraction' messages (should be skipped)")
        print("  2. Verify GPU_PERSISTENT_ROUTER=1 is active")
        print("  3. Profile with NVPROF: nvprof python main.py --test-manhattan")
        print("  4. Check GPU utilization with: watch -n 0.1 nvidia-smi")

    # Recommendations for further optimization
    if 10 <= speedup < 127:
        print("\n┌─────────────────────────────────────────────────────────────────┐")
        print("│              RECOMMENDATIONS FOR FURTHER OPTIMIZATION           │")
        print("├─────────────────────────────────────────────────────────────────┤")
        print("│ Current status: Good speedup achieved, but not at theoretical max│")
        print("│                                                                 │")
        print("│ Next optimization opportunities (Weekend Plan Phases 6-9):      │")
        print("│ • Phase 6: Status logging without host stalls (pinned memory)   │")
        print("│ • Phase 7: Device-side parent→edge mapping (cuckoo hashing)    │")
        print("│ • Phase 8: Batch scheduler on device (dynamic SM utilization)   │")
        print("│ • Phase 9: Clean feature-flag glue (production polish)          │")
        print("│                                                                 │")
        print(f"│ Potential additional gain: {(127-speedup)/speedup*100:.0f}% (up to {127:.0f}× total)         │")
        print("└─────────────────────────────────────────────────────────────────┘")

def main():
    """Main benchmark execution."""
    print_header("GPU ROUTER PERFORMANCE BENCHMARK")
    print("Objective: Validate 10-127× speedup goals from Weekend Plan")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Phase 1: Baseline (CPU) measurement
    print_header("PHASE 1: BASELINE (CPU) MEASUREMENT")
    baseline_env = {
        'GPU_PERSISTENT_ROUTER': '0'  # Disable GPU optimizations
    }
    baseline_metrics = run_test_with_timing(baseline_env, "BASELINE")

    if baseline_metrics:
        analyze_gpu_utilization("BASELINE")

    # Phase 2: GPU persistent router measurement
    print_header("PHASE 2: GPU PERSISTENT ROUTER MEASUREMENT")
    gpu_env = {
        'GPU_PERSISTENT_ROUTER': '1'  # Enable GPU optimizations
    }
    gpu_metrics = run_test_with_timing(gpu_env, "GPU")

    if gpu_metrics:
        analyze_gpu_utilization("GPU")

    # Generate report
    generate_report(baseline_metrics, gpu_metrics)

    # Save metrics to file
    print("\n" + "="*80)
    print("Saving detailed metrics...")

    with open('benchmark_results.txt', 'w') as f:
        f.write("GPU ROUTER PERFORMANCE BENCHMARK RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Benchmark Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("BASELINE (CPU):\n")
        if baseline_metrics:
            for key, val in baseline_metrics.items():
                f.write(f"  {key}: {val}\n")
        else:
            f.write("  FAILED\n")

        f.write("\nGPU PERSISTENT:\n")
        if gpu_metrics:
            for key, val in gpu_metrics.items():
                f.write(f"  {key}: {val}\n")
        else:
            f.write("  FAILED\n")

        if baseline_metrics and gpu_metrics and baseline_metrics['success'] and gpu_metrics['success']:
            speedup = baseline_metrics['elapsed_sec'] / gpu_metrics['elapsed_sec']
            f.write(f"\nSPEEDUP: {speedup:.1f}×\n")
            f.write(f"GOAL ACHIEVED: {'YES' if speedup >= 10 else 'NO'}\n")

    print("Detailed metrics saved to: benchmark_results.txt")
    print("="*80)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
        sys.exit(130)
    except Exception as e:
        logging.error(f"Benchmark failed: {e}", exc_info=True)
        sys.exit(1)
