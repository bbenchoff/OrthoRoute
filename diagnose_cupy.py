#!/usr/bin/env python
"""Diagnostic script to check CuPy capabilities and GPU state."""

import cupy as cp
import numpy as np

print("=" * 80)
print("CuPy and GPU Diagnostic Report")
print("=" * 80)

# CuPy version
print(f"\n1. CuPy Version: {cp.__version__}")

# GPU info
print("\n2. GPU Information:")
device = cp.cuda.Device()
print(f"   Device ID: {device.id}")
print(f"   Device Name: {cp.cuda.runtime.getDeviceProperties(device.id)['name'].decode()}")
print(f"   Compute Capability: {device.compute_capability}")

# Memory info
free, total = device.mem_info
print(f"\n3. GPU Memory:")
print(f"   Free: {free / 1e9:.2f} GB")
print(f"   Total: {total / 1e9:.2f} GB")
print(f"   Used: {(total - free) / 1e9:.2f} GB")

# Memory pool
mempool = cp.get_default_memory_pool()
print(f"\n4. Memory Pool:")
print(f"   Pool used: {mempool.used_bytes() / 1e9:.2f} GB")
print(f"   Pool total: {mempool.total_bytes() / 1e9:.2f} GB")
print(f"   Pool limit: {mempool.get_limit() / 1e9:.2f} GB" if mempool.get_limit() > 0 else "   Pool limit: No limit")

# Test unpackbits
print("\n5. Testing cp.unpackbits():")
try:
    # Test without axis (should work)
    test_bits = cp.array([0b10101010], dtype=cp.uint8)
    unpacked = cp.unpackbits(test_bits, bitorder='little')
    print(f"   [OK] unpackbits(1D) works: {unpacked}")

    # Test with axis (might not work)
    test_bits_2d = cp.array([[0b10101010], [0b11001100]], dtype=cp.uint8)
    try:
        unpacked_axis = cp.unpackbits(test_bits_2d, axis=1, bitorder='little')
        print(f"   [OK] unpackbits(2D, axis=1) works: shape={unpacked_axis.shape}")
    except NotImplementedError:
        print(f"   [FAIL] unpackbits(2D, axis=1) NOT supported")
        print(f"   → Workaround: ravel(), unpackbits(), reshape()")
        unpacked_manual = cp.unpackbits(test_bits_2d.ravel(), bitorder='little').reshape(2, -1)
        print(f"   [OK] Workaround works: shape={unpacked_manual.shape}")

except Exception as e:
    print(f"   [FAIL] Error: {e}")

# Test stamp array dtypes
print("\n6. Testing uint16 arrays:")
try:
    test_stamp = cp.zeros(100, dtype=cp.uint16)
    test_stamp[50] = 1234
    print(f"   [OK] uint16 arrays work: {test_stamp[50]}")
except Exception as e:
    print(f"   [FAIL] Error: {e}")

# Test bitset operations
print("\n7. Testing bitset operations:")
try:
    N = 1000
    bits = cp.zeros((N + 7) // 8, dtype=cp.uint8)

    # Set bit 500
    idx = 500
    bits[idx >> 3] |= (1 << (idx & 7))

    # Check bit 500
    is_set = (bits[idx >> 3] >> (idx & 7)) & 1
    print(f"   [OK] Bitset operations work: bit {idx} is_set={is_set}")
except Exception as e:
    print(f"   [FAIL] Error: {e}")

# Check for GPU errors
print("\n8. Checking GPU Error State:")
try:
    err = cp.cuda.runtime.getLastError()
    if err == 0:
        print(f"   [OK] No pending CUDA errors")
    else:
        err_name = cp.cuda.runtime.getErrorName(err).decode()
        err_string = cp.cuda.runtime.getErrorString(err).decode()
        print(f"   [FAIL] CUDA Error {err}: {err_name} - {err_string}")
        print(f"   → GPU needs reset (close all Python processes)")
except Exception as e:
    print(f"   Warning: {e}")

# Try a simple kernel
print("\n9. Testing Simple CUDA Kernel:")
try:
    simple_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void test_kernel(float* out, int N) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < N) {
            out[tid] = tid * 2.0f;
        }
    }
    ''', 'test_kernel')

    test_arr = cp.zeros(1000, dtype=cp.float32)
    simple_kernel((4,), (256,), (test_arr, 1000))
    cp.cuda.Stream.null.synchronize()

    if test_arr[10] == 20.0:
        print(f"   [OK] Simple kernel works: test_arr[10]={test_arr[10]}")
    else:
        print(f"   [FAIL] Simple kernel failed: expected 20.0, got {test_arr[10]}")
except Exception as e:
    print(f"   [FAIL] Kernel launch failed: {e}")
    print(f"   → GPU may be in error state, needs reset")

print("\n" + "=" * 80)
print("Diagnostic Complete")
print("=" * 80)
