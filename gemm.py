import os

num_threads = '1'
os.environ["OMP_NUM_THREADS"] = num_threads # OpenMP
os.environ["OPENBLAS_NUM_THREADS"] = num_threads # OpenBLAS
os.environ["MKL_NUM_THREADS"] = num_threads # Intel MKL
os.environ["VECLIB_MAXIMUM_THREADS"] = num_threads
os.environ["NUMEXPR_NUM_THREADS"] = num_threads

import numpy as np
import time

# how to interpret a vector?
# for example vec = (1, 1)
# how to draw it? where is it in a xy space?
# default xy space have x at (1, 0) and y at (0, 1)
# so to find position of vec we need x * vec.x + y * vec.y
#
# so space is defined by x and y vectors and any vector in that space is:
#
# hence we can use matrices for definition of space where matrix is [x, y] where x and y are column vectors
# space = x.x y.x
#         x.y y.y
# (vec.x * x.x + vec.y * y.x, vec.x * x.y + vec.y * y.y) = (dot(vec, space[0]), dot(vec, space[1]))
#
# this is about mat * vec, but what is mat * mat?
# first mat is defining space, second mat is packed column vectors? if so then
# m0 * m1 = [m0 * m1[0], m0 * m1[1]]
# m0 * m1 = [dot(m0[0], m1[0]), dot(m0[0], m1[1]),
#            dot(m0[1], m1[0]), dot(m0[1], m1[1])]
# how much flop in matrix vec multiply? its space.height * space.width since each dot have vec_n flop and we have space.height dots
# vec_length * space.height
# how much flop in matrix matrix multiple? its m1.height * m0.height * m1.width = k * m * n

M = 2048
N = 2048
P = 2048

flops = 2 * M * N * P

print(f"GFLOP {flops / 1e9:.3f}")

A = np.random.randn(M, N).astype(np.float32)
B = np.random.randn(N, P).astype(np.float32)

ITERS = 10
total_time = 0.0

for i in range(ITERS):
    st = time.monotonic()
    C = A @ B
    et = time.monotonic()
    tt = et - st
    total_time += tt

print(f"numpy AVG GFLOPS {flops / (total_time / ITERS) * 1e-9:.3f}")
