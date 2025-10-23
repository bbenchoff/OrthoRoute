#!/usr/bin/env python3
"""
Profiling Test Runner for OrthoRoute
Runs routing test with detailed timing and correctness tracking
"""

import subprocess
import time
import re
import sys
import json
from datetime import datetime
from pathlib import Path

def run_baseline_test(duration_sec=600, log_file='baseline_test.log'):
    """Run baseline routing test with timing"""
    print(f"Running baseline test for {duration_sec}s...")
    print(f"Log: {log_file}")

    start_time = time.time()

    # Start routing test (use shell to preserve environment)
    import os
    env = os.environ.copy()
    env['ORTHO_CPU_ONLY'] = '1'  # CPU-only for baseline

    process = subprocess.Popen(
        'python main.py --test-manhattan',
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        shell=True,
        env=env
    )

    metrics = {
        'start_time': datetime.now().isoformat(),
        'iterations': [],
        'timings': {},
        'correctness': {}
    }

    iteration_data = {}
    current_iter = 0

    with open(log_file, 'w') as f:
        try:
            for line in iter(process.stdout.readline, ''):
                if not line:
                    break

                f.write(line)
                f.flush()

                # Check timeout
                if time.time() - start_time > duration_sec:
                    print(f"\nTimeout reached ({duration_sec}s), terminating...")
                    process.terminate()
                    break

                # Parse iteration results
                iter_match = re.search(r'\[ITER (\d+)\] routed=(\d+) failed=(\d+) overuse=([\d.]+)', line)
                if iter_match:
                    iter_num = int(iter_match.group(1))
                    current_iter = iter_num
                    iteration_data = {
                        'iteration': iter_num,
                        'routed': int(iter_match.group(2)),
                        'failed': int(iter_match.group(3)),
                        'overuse': float(iter_match.group(4)),
                        'timestamp': time.time() - start_time
                    }
                    metrics['iterations'].append(iteration_data)
                    print(f"Iter {iter_num}: {iteration_data['routed']} routed, "
                          f"{iteration_data['failed']} failed, overuse={iteration_data['overuse']}")

                # Parse timing info
                if 'took' in line.lower() and 'ms' in line.lower():
                    time_match = re.search(r'(\w+).*?(\d+\.?\d*)\s*ms', line)
                    if time_match:
                        stage = time_match.group(1)
                        ms = float(time_match.group(2))
                        if stage not in metrics['timings']:
                            metrics['timings'][stage] = []
                        metrics['timings'][stage].append(ms)

        except KeyboardInterrupt:
            print("\nInterrupted by user")
            process.terminate()
        finally:
            process.wait()

    metrics['end_time'] = datetime.now().isoformat()
    metrics['duration_sec'] = time.time() - start_time

    # Calculate summary
    if metrics['iterations']:
        final_iter = metrics['iterations'][-1]
        metrics['correctness'] = {
            'total_iterations': len(metrics['iterations']),
            'final_routed': final_iter['routed'],
            'final_failed': final_iter['failed'],
            'final_overuse': final_iter['overuse'],
            'convergence_rate': calculate_convergence_rate(metrics['iterations'])
        }

    # Save metrics
    metrics_file = log_file.replace('.log', '_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nBaseline test complete!")
    print(f"  Duration: {metrics['duration_sec']:.1f}s")
    print(f"  Iterations: {len(metrics['iterations'])}")
    if metrics['correctness']:
        print(f"  Final routed: {metrics['correctness']['final_routed']}")
        print(f"  Final overuse: {metrics['correctness']['final_overuse']}")

    return metrics

def calculate_convergence_rate(iterations):
    """Calculate how quickly overuse converges"""
    if len(iterations) < 2:
        return 0.0

    # Calculate exponential decay rate
    overuses = [it['overuse'] for it in iterations if it['overuse'] > 0]
    if len(overuses) < 2:
        return 0.0

    # Simple linear regression on log(overuse)
    import math
    log_overuses = [math.log(o + 1) for o in overuses]
    avg_decay = (log_overuses[0] - log_overuses[-1]) / len(log_overuses)
    return avg_decay

if __name__ == '__main__':
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    log_file = sys.argv[2] if len(sys.argv) > 2 else 'baseline_test.log'
    run_baseline_test(duration, log_file)
