#!/usr/bin/env python3
"""
GPU Dot Product Benchmark using PyTorch on M4 Max (MPS backend)
Reference performance for dot product with N = 1 << 22 floats
"""

import torch
import time

# Configuration
N = 1 << 22  # 4,194,304 floats
ITERS = 100

print(f"PyTorch MPS Dot Product Benchmark")
print(f"N = {N:,} floats")

# Generate random vectors
x = torch.randn(N, dtype=torch.float32, device='mps')
y = torch.randn(N, dtype=torch.float32, device='mps')

# Calculate FLOPS (2N operations: N multiplies + N adds)
flops = N * 2
print(f"TOTAL MFLOP: {flops / 1e6:.3f}\n")

# Warmup
for _ in range(10):
    result = torch.dot(x, y)
    torch.mps.synchronize()

# Benchmark
total_time = 0.0
for i in range(ITERS):
    start = time.perf_counter()
    result = torch.dot(x, y)
    torch.mps.synchronize()
    elapsed = time.perf_counter() - start
    total_time += elapsed

avg_time = total_time / ITERS
avg_gflops = (flops * 1e-9) / avg_time

print(f"PyTorch MPS AVG GFLOPS: {avg_gflops:.3f}")
print(f"\nResult: {result.item():.6f}")
