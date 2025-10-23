#!/usr/bin/env python3
"""
CUDA Kernel Optimization Script
Systematically optimizes CUDA kernels with correctness verification
"""

import subprocess
import json
import time
from pathlib import Path

class CUDAOptimizer:
    def __init__(self):
        self.baseline_metrics = None
        self.optimizations = []

    def run_test(self, test_name, duration_sec=120):
        """Run routing test and extract metrics"""
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print(f"{'='*60}")

        log_file = f"test_{test_name}.log"
        start = time.time()

        try:
            result = subprocess.run(
                ['python', 'main.py', '--test-manhattan'],
                capture_output=True,
                text=True,
                timeout=duration_sec,
                env={'ORTHO_CPU_ONLY': '0'}  # GPU mode
            )
            output = result.stdout
        except subprocess.TimeoutExpired as e:
            output = e.stdout.decode() if e.stdout else ""

        elapsed = time.time() - start

        # Save log
        with open(log_file, 'w') as f:
            f.write(output)

        # Extract metrics
        metrics = self.parse_metrics(output, elapsed)
        metrics['test_name'] = test_name
        metrics['log_file'] = log_file

        return metrics

    def parse_metrics(self, output, elapsed_sec):
        """Extract iteration metrics from log"""
        import re

        metrics = {
            'elapsed_sec': elapsed_sec,
            'iterations': [],
            'total_nets_routed': 0
        }

        # Find iteration lines
        for line in output.split('\n'):
            match = re.search(r'\[ITER (\d+)\] routed=(\d+) failed=(\d+) overuse=([\d.]+)', line)
            if match:
                iter_data = {
                    'iteration': int(match.group(1)),
                    'routed': int(match.group(2)),
                    'failed': int(match.group(3)),
                    'overuse': float(match.group(4))
                }
                metrics['iterations'].append(iter_data)

        if metrics['iterations']:
            metrics['total_nets_routed'] = metrics['iterations'][-1]['routed']

        return metrics

    def verify_correctness(self, metrics):
        """Verify optimization maintains correctness"""
        if not self.baseline_metrics:
            print("ERROR: No baseline metrics to compare against!")
            return False

        baseline_iters = {i['iteration']: i for i in self.baseline_metrics['iterations']}
        test_iters = {i['iteration']: i for i in metrics['iterations']}

        print(f"\nCorrectness Verification:")
        print(f"{'Iter':<6} {'Baseline':<10} {'Optimized':<10} {'Status':<10}")
        print("-" * 40)

        all_good = True
        for iter_num in sorted(baseline_iters.keys()):
            if iter_num not in test_iters:
                print(f"{iter_num:<6} {'N/A':<10} {'MISSING':<10} {'FAIL':<10}")
                all_good = False
                continue

            baseline_routed = baseline_iters[iter_num]['routed']
            test_routed = test_iters[iter_num]['routed']

            status = "PASS" if test_routed >= baseline_routed else "FAIL"
            if status == "FAIL":
                all_good = False

            print(f"{iter_num:<6} {baseline_routed:<10} {test_routed:<10} {status:<10}")

        return all_good

    def calculate_speedup(self, metrics):
        """Calculate speedup vs baseline"""
        if not self.baseline_metrics:
            return 0.0

        baseline_time = self.baseline_metrics['elapsed_sec']
        test_time = metrics['elapsed_sec']

        return baseline_time / test_time if test_time > 0 else 0.0

    def run_optimization_cycle(self, opt_name, opt_description):
        """Run one optimization cycle with verification"""
        print(f"\n{'#'*60}")
        print(f"OPTIMIZATION: {opt_name}")
        print(f"Description: {opt_description}")
        print(f"{'#'*60}")

        # Run test
        metrics = self.run_test(opt_name, duration_sec=180)

        # Verify correctness
        correct = self.verify_correctness(metrics)

        # Calculate speedup
        speedup = self.calculate_speedup(metrics)

        result = {
            'name': opt_name,
            'description': opt_description,
            'correct': correct,
            'speedup': speedup,
            'metrics': metrics
        }

        self.optimizations.append(result)

        print(f"\nResults:")
        print(f"  Correctness: {'PASS' if correct else 'FAIL'}")
        print(f"  Speedup: {speedup:.2f}x")

        if not correct:
            print("\n WARNING: Correctness check FAILED - consider reverting")

        return result

    def generate_report(self):
        """Generate optimization report"""
        report = []
        report.append("# CUDA Optimization Results\n")
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        if self.baseline_metrics:
            report.append("## Baseline\n")
            report.append(f"- Duration: {self.baseline_metrics['elapsed_sec']:.1f}s\n")
            report.append(f"- Iterations: {len(self.baseline_metrics['iterations'])}\n")
            report.append(f"- Total Routed: {self.baseline_metrics['total_nets_routed']}\n\n")

        report.append("## Optimizations\n\n")
        for opt in self.optimizations:
            report.append(f"### {opt['name']}\n")
            report.append(f"**Description**: {opt['description']}\n\n")
            report.append(f"**Results**:\n")
            report.append(f"- Correctness: {'✓ PASS' if opt['correct'] else '✗ FAIL'}\n")
            report.append(f"- Speedup: {opt['speedup']:.2f}x\n")
            report.append(f"- Duration: {opt['metrics']['elapsed_sec']:.1f}s\n\n")

        # Best result
        valid_opts = [o for o in self.optimizations if o['correct']]
        if valid_opts:
            best = max(valid_opts, key=lambda x: x['speedup'])
            report.append(f"## Best Result\n")
            report.append(f"**{best['name']}**: {best['speedup']:.2f}x speedup\n\n")

        return ''.join(report)

if __name__ == '__main__':
    optimizer = CUDAOptimizer()

    # Load baseline
    print("Loading baseline metrics...")
    # TODO: Load from baseline test results

    print("\nStarting optimization cycles...")
    # Optimizations will be added here

    # Generate report
    report = optimizer.generate_report()
    with open('CUDA_OPTIMIZATION_REPORT.md', 'w') as f:
        f.write(report)

    print("\n" + "="*60)
    print("Optimization complete! See CUDA_OPTIMIZATION_REPORT.md")
    print("="*60)
