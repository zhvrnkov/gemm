#include <metal_stdlib>
using namespace metal;

kernel void sgemm(
                  const device float* A,
                  const device float* B,
                  device float* C,
                  constant const uint64_t& N,
//                  threadgroup float* As,
//                  threadgroup float* Bs,
                  uint2 gid [[thread_position_in_grid]],
                  uint2 lid [[thread_position_in_threadgroup]],
                  uint2 group_id [[threadgroup_position_in_grid]],
                  uint2 group_size [[threads_per_threadgroup]]
                  )
{
    constexpr uint64_t dK = 16;
    threadgroup float As[dK * dK];
    threadgroup float Bs[dK * dK];

    device float* bC = &C[group_id.y * dK * N + group_id.x * dK];
    const device float* bA = &A[group_id.y * dK * N];
    const device float* bB = &B[group_id.x * dK];
    
    simdgroup_float8x8 Ams[2][2];
    simdgroup_float8x8 Bms[2][2];
    simdgroup_float8x8 Rms[2][2];
    for (int i = 0; i < 4; i++) Rms[i / 2][i % 2] = simdgroup_float8x8(0);

    for (uint64_t bk = 0; bk < N; bk += dK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint2 offset = uint2(0, 0);
        As[(lid.y + offset.y) * dK + (lid.x + offset.x)] = bA[bk + (lid.y + offset.y) * N + (lid.x + offset.x)];
        Bs[(lid.y + offset.y) * dK + (lid.x + offset.x)] = bB[bk * N + (lid.y + offset.y) * N + (lid.x + offset.x)];
        offset = uint2(8, 0);
        As[(lid.y + offset.y) * dK + (lid.x + offset.x)] = bA[bk + (lid.y + offset.y) * N + (lid.x + offset.x)];
        Bs[(lid.y + offset.y) * dK + (lid.x + offset.x)] = bB[bk * N + (lid.y + offset.y) * N + (lid.x + offset.x)];
        offset = uint2(0, 8);
        As[(lid.y + offset.y) * dK + (lid.x + offset.x)] = bA[bk + (lid.y + offset.y) * N + (lid.x + offset.x)];
        Bs[(lid.y + offset.y) * dK + (lid.x + offset.x)] = bB[bk * N + (lid.y + offset.y) * N + (lid.x + offset.x)];
        offset = uint2(8, 8);
        As[(lid.y + offset.y) * dK + (lid.x + offset.x)] = bA[bk + (lid.y + offset.y) * N + (lid.x + offset.x)];
        Bs[(lid.y + offset.y) * dK + (lid.x + offset.x)] = bB[bk * N + (lid.y + offset.y) * N + (lid.x + offset.x)];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        simdgroup_load(Ams[0][0], &As[0 * 8 * dK + 0 * 8], dK);
        simdgroup_load(Bms[0][0], &Bs[0 * 8 * dK + 0 * 8], dK);
        
        simdgroup_load(Ams[0][1], &As[0 * 8 * dK + 1 * 8], dK);
        simdgroup_load(Bms[0][1], &Bs[0 * 8 * dK + 1 * 8], dK);
        
        simdgroup_load(Ams[1][0], &As[1 * 8 * dK + 0 * 8], dK);
        simdgroup_load(Bms[1][0], &Bs[1 * 8 * dK + 0 * 8], dK);
        
        simdgroup_load(Ams[1][1], &As[1 * 8 * dK + 1 * 8], dK);
        simdgroup_load(Bms[1][1], &Bs[1 * 8 * dK + 1 * 8], dK);
        
        simdgroup_multiply_accumulate(Rms[0][0], Ams[0][0], Bms[0][0], Rms[0][0]);
        simdgroup_multiply_accumulate(Rms[0][0], Ams[0][1], Bms[1][0], Rms[0][0]);
        
        simdgroup_multiply_accumulate(Rms[0][1], Ams[0][0], Bms[0][1], Rms[0][1]);
        simdgroup_multiply_accumulate(Rms[0][1], Ams[0][1], Bms[1][1], Rms[0][1]);
        
        simdgroup_multiply_accumulate(Rms[1][0], Ams[1][0], Bms[0][0], Rms[1][0]);
        simdgroup_multiply_accumulate(Rms[1][0], Ams[1][1], Bms[1][0], Rms[1][0]);
        
        simdgroup_multiply_accumulate(Rms[1][1], Ams[1][0], Bms[0][1], Rms[1][1]);
        simdgroup_multiply_accumulate(Rms[1][1], Ams[1][1], Bms[1][1], Rms[1][1]);
    }

    simdgroup_store(Rms[0][0], &bC[0 * N + 0], N);
    simdgroup_store(Rms[0][1], &bC[0 * N + 8], N);
    simdgroup_store(Rms[1][0], &bC[8 * N + 0], N);
    simdgroup_store(Rms[1][1], &bC[8 * N + 8], N);
}

kernel void simd_test(
                      const device float* A,
                      const device float* B,
                      device float* C,
                      uint gid [[thread_position_in_grid]],
                      uint lid [[thread_position_in_threadgroup]],
                      uint tgp_size [[threads_per_threadgroup]],
                      uint sid [[simdgroup_index_in_threadgroup]],
                      uint sgp_size [[threads_per_simdgroup]]
                      )
{
    simdgroup_float8x8 Am;
    simdgroup_float8x8 Bm;
    simdgroup_float8x8 Cm;
    
    simdgroup_load(Am, A);
    simdgroup_load(Bm, B);
    simdgroup_multiply(Cm, Am, Bm);

    simdgroup_store(Cm, C);
}

kernel void sgemm_32x32(
                        const device float* A,
                        const device float* B,
                        device float* C,
                        constant const uint64_t& N,
                        //                  threadgroup float* As,
                        //                  threadgroup float* Bs,
                        uint2 gid [[thread_position_in_grid]],
                        uint2 lid [[thread_position_in_threadgroup]],
                        uint2 group_id [[threadgroup_position_in_grid]],
                        uint2 group_size [[threads_per_threadgroup]]
                        )
{
    constexpr auto dim = 4;
    constexpr auto dK = dim * 8;
    // threadgroup float As[dK * dK];
    // threadgroup float Bs[dK * dK];

    // C += lid.y * 16 * N;
    A += group_id.y * (dK * 2) * N;
    B += group_id.x * dK;
    C += group_id.y * (dK * 2) * N + group_id.x * dK;

    A += lid.y * dK * N;
    C += lid.y * dK * N;
    
    simdgroup_float8x8 Am[dim];
    simdgroup_float8x8 Bm[dim];
    simdgroup_float8x8 acc[dim][dim];
    for (int y = 0; y < dim; y++) {
        for (int x = 0; x < dim; x++) {
            acc[y][x] = simdgroup_float8x8(0);
        }
    }

    for (uint k = 0; k < N; k += 8) {
      for (int x = 0; x < dim; x++) {
        simdgroup_load(Am[x], A + (k) + (x * 8 * N), N);
      }

      for (int y = 0; y < dim; y++) {
        simdgroup_load(Bm[y], B + (k * N) + (y * 8), N);
      }

      for (int y = 0; y < dim; y++) {
        for (int x = 0; x < dim; x++) {
          simdgroup_multiply_accumulate(acc[y][x], Am[y], Bm[x], acc[y][x]);
        }
      }
    }

    for (int y = 0; y < dim; y++) {
        for (int x = 0; x < dim; x++) {
            simdgroup_store(acc[y][x], C + (y * 8 * N) + (x * 8), N);
        }
    }
}


kernel void sgemm_32x32_unrolled(
                        const device float* A,
                        const device float* B,
                        device float* C,
                        constant const uint64_t& N,
                        //                  threadgroup float* As,
                        //                  threadgroup float* Bs,
                        uint2 gid [[thread_position_in_grid]],
                        uint2 lid [[thread_position_in_threadgroup]],
                        uint2 group_id [[threadgroup_position_in_grid]],
                        uint2 group_size [[threads_per_threadgroup]]
                        )
{
    constexpr auto dim = 4;
    constexpr auto dK = dim * 8;
    // threadgroup float As[dK * dK];
    // threadgroup float Bs[dK * dK];

    // C += lid.y * 16 * N;
    A += group_id.y * (dK * 2) * N;
    B += group_id.x * dK;
    C += group_id.y * (dK * 2) * N + group_id.x * dK;

    A += lid.y * dK * N;
    C += lid.y * dK * N;
    
    simdgroup_float8x8 Am[dim];
    simdgroup_float8x8 Bm[dim];
    simdgroup_float8x8 acc[dim][dim];

    // --- Unrolled initialization of acc matrix ---
    acc[0][0] = simdgroup_float8x8(0);
    acc[0][1] = simdgroup_float8x8(0);
    acc[0][2] = simdgroup_float8x8(0);
    acc[0][3] = simdgroup_float8x8(0);

    acc[1][0] = simdgroup_float8x8(0);
    acc[1][1] = simdgroup_float8x8(0);
    acc[1][2] = simdgroup_float8x8(0);
    acc[1][3] = simdgroup_float8x8(0);

    acc[2][0] = simdgroup_float8x8(0);
    acc[2][1] = simdgroup_float8x8(0);
    acc[2][2] = simdgroup_float8x8(0);
    acc[2][3] = simdgroup_float8x8(0);

    acc[3][0] = simdgroup_float8x8(0);
    acc[3][1] = simdgroup_float8x8(0);
    acc[3][2] = simdgroup_float8x8(0);
    acc[3][3] = simdgroup_float8x8(0);

    for (uint k = 0; k < N; k += 8) {
        // --- Unrolled loads from A into Am[0..3] ---
        simdgroup_load(Am[0], A + (k) + (0 * 8 * N), N);
        simdgroup_load(Am[1], A + (k) + (1 * 8 * N), N);
        simdgroup_load(Am[2], A + (k) + (2 * 8 * N), N);
        simdgroup_load(Am[3], A + (k) + (3 * 8 * N), N);

        // --- Unrolled loads from B into Bm[0..3] ---
        simdgroup_load(Bm[0], B + (k * N) + (0 * 8), N);
        simdgroup_load(Bm[1], B + (k * N) + (1 * 8), N);
        simdgroup_load(Bm[2], B + (k * N) + (2 * 8), N);
        simdgroup_load(Bm[3], B + (k * N) + (3 * 8), N);

        // --- Unrolled multiply-accumulate for all 4x4 combinations ---
        // y = 0
        simdgroup_multiply_accumulate(acc[0][0], Am[0], Bm[0], acc[0][0]);
        simdgroup_multiply_accumulate(acc[0][1], Am[0], Bm[1], acc[0][1]);
        simdgroup_multiply_accumulate(acc[0][2], Am[0], Bm[2], acc[0][2]);
        simdgroup_multiply_accumulate(acc[0][3], Am[0], Bm[3], acc[0][3]);

        // y = 1
        simdgroup_multiply_accumulate(acc[1][0], Am[1], Bm[0], acc[1][0]);
        simdgroup_multiply_accumulate(acc[1][1], Am[1], Bm[1], acc[1][1]);
        simdgroup_multiply_accumulate(acc[1][2], Am[1], Bm[2], acc[1][2]);
        simdgroup_multiply_accumulate(acc[1][3], Am[1], Bm[3], acc[1][3]);

        // y = 2
        simdgroup_multiply_accumulate(acc[2][0], Am[2], Bm[0], acc[2][0]);
        simdgroup_multiply_accumulate(acc[2][1], Am[2], Bm[1], acc[2][1]);
        simdgroup_multiply_accumulate(acc[2][2], Am[2], Bm[2], acc[2][2]);
        simdgroup_multiply_accumulate(acc[2][3], Am[2], Bm[3], acc[2][3]);

        // y = 3
        simdgroup_multiply_accumulate(acc[3][0], Am[3], Bm[0], acc[3][0]);
        simdgroup_multiply_accumulate(acc[3][1], Am[3], Bm[1], acc[3][1]);
        simdgroup_multiply_accumulate(acc[3][2], Am[3], Bm[2], acc[3][2]);
        simdgroup_multiply_accumulate(acc[3][3], Am[3], Bm[3], acc[3][3]);
    }

    // --- Unrolled stores from acc to C ---
    simdgroup_store(acc[0][0], C + (0 * 8 * N) + (0 * 8), N);
    simdgroup_store(acc[0][1], C + (0 * 8 * N) + (1 * 8), N);
    simdgroup_store(acc[0][2], C + (0 * 8 * N) + (2 * 8), N);
    simdgroup_store(acc[0][3], C + (0 * 8 * N) + (3 * 8), N);

    simdgroup_store(acc[1][0], C + (1 * 8 * N) + (0 * 8), N);
    simdgroup_store(acc[1][1], C + (1 * 8 * N) + (1 * 8), N);
    simdgroup_store(acc[1][2], C + (1 * 8 * N) + (2 * 8), N);
    simdgroup_store(acc[1][3], C + (1 * 8 * N) + (3 * 8), N);

    simdgroup_store(acc[2][0], C + (2 * 8 * N) + (0 * 8), N);
    simdgroup_store(acc[2][1], C + (2 * 8 * N) + (1 * 8), N);
    simdgroup_store(acc[2][2], C + (2 * 8 * N) + (2 * 8), N);
    simdgroup_store(acc[2][3], C + (2 * 8 * N) + (3 * 8), N);

    simdgroup_store(acc[3][0], C + (3 * 8 * N) + (0 * 8), N);
    simdgroup_store(acc[3][1], C + (3 * 8 * N) + (1 * 8), N);
    simdgroup_store(acc[3][2], C + (3 * 8 * N) + (2 * 8), N);
    simdgroup_store(acc[3][3], C + (3 * 8 * N) + (3 * 8), N);
}

