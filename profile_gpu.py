#!/usr/bin/env python3
"""
GPU Profiling Monitor for OrthoRoute
Samples nvidia-smi every 500ms and logs GPU utilization, memory, temperature
"""

import subprocess
import time
import sys
import re
from datetime import datetime

def parse_nvidia_smi_line(line):
    """Parse nvidia-smi output line for GPU stats"""
    # Example: |   0  NVIDIA GeForce...  | 85%   45C   P2   120W / 250W |   4096MiB /  8192MiB |
    match = re.search(r'\|\s+\d+\s+.*?\|\s+(\d+)%\s+(\d+)C\s+.*?\|\s+(\d+)MiB\s+/\s+(\d+)MiB', line)
    if match:
        util = int(match.group(1))
        temp = int(match.group(2))
        mem_used = int(match.group(3))
        mem_total = int(match.group(4))
        return {
            'util': util,
            'temp': temp,
            'mem_used': mem_used,
            'mem_total': mem_total,
            'mem_pct': (mem_used / mem_total * 100) if mem_total > 0 else 0
        }
    return None

def monitor_gpu(duration_sec=600, interval_sec=0.5, output_file='gpu_profile.log'):
    """Monitor GPU for specified duration"""
    print(f"Starting GPU monitoring for {duration_sec}s (interval={interval_sec}s)")
    print(f"Output: {output_file}")

    start_time = time.time()
    samples = []

    with open(output_file, 'w') as f:
        f.write("timestamp,util_pct,temp_c,mem_used_mb,mem_total_mb,mem_pct\n")

        while time.time() - start_time < duration_sec:
            try:
                # Run nvidia-smi
                result = subprocess.run(
                    ['nvidia-smi'],
                    capture_output=True,
                    text=True,
                    timeout=2
                )

                # Parse output
                for line in result.stdout.split('\n'):
                    stats = parse_nvidia_smi_line(line)
                    if stats:
                        timestamp = time.time() - start_time
                        f.write(f"{timestamp:.3f},{stats['util']},{stats['temp']},"
                               f"{stats['mem_used']},{stats['mem_total']},{stats['mem_pct']:.1f}\n")
                        f.flush()

                        samples.append(stats)

                        # Print periodic summary
                        if len(samples) % 20 == 0:
                            avg_util = sum(s['util'] for s in samples[-20:]) / 20
                            print(f"[{timestamp:.1f}s] GPU: {stats['util']}% (avg {avg_util:.1f}%), "
                                  f"Mem: {stats['mem_pct']:.1f}%, Temp: {stats['temp']}C")
                        break

                time.sleep(interval_sec)

            except subprocess.TimeoutExpired:
                print("WARNING: nvidia-smi timeout")
                time.sleep(interval_sec)
            except FileNotFoundError:
                print("ERROR: nvidia-smi not found - GPU monitoring disabled")
                return
            except Exception as e:
                print(f"ERROR: {e}")
                time.sleep(interval_sec)

    # Print summary
    if samples:
        avg_util = sum(s['util'] for s in samples) / len(samples)
        max_util = max(s['util'] for s in samples)
        avg_mem = sum(s['mem_pct'] for s in samples) / len(samples)
        print(f"\nGPU Monitoring Summary:")
        print(f"  Samples: {len(samples)}")
        print(f"  Avg Utilization: {avg_util:.1f}%")
        print(f"  Max Utilization: {max_util}%")
        print(f"  Avg Memory: {avg_mem:.1f}%")

if __name__ == '__main__':
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    output = sys.argv[2] if len(sys.argv) > 2 else 'gpu_profile.log'
    monitor_gpu(duration, output_file=output)
