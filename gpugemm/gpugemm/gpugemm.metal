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
    
    threadgroup float Rmfs[2][2][8 * 8];
    simdgroup_store(Rms[0][0], Rmfs[0][0]);
    simdgroup_store(Rms[0][1], Rmfs[0][1]);
    simdgroup_store(Rms[1][0], Rmfs[1][0]);
    simdgroup_store(Rms[1][1], Rmfs[1][1]);

    uint2 offset = uint2(0);
    bC[(lid.y + offset.y * 8) * N + (lid.x + offset.x * 8)] = Rmfs[offset.y][offset.x][lid.y * 8 + lid.x];
    offset = uint2(1, 0);
    bC[(lid.y + offset.y * 8) * N + (lid.x + offset.x * 8)] = Rmfs[offset.y][offset.x][lid.y * 8 + lid.x];
    offset = uint2(0, 1);
    bC[(lid.y + offset.y * 8) * N + (lid.x + offset.x * 8)] = Rmfs[offset.y][offset.x][lid.y * 8 + lid.x];
    offset = uint2(1, 1);
    bC[(lid.y + offset.y * 8) * N + (lid.x + offset.x * 8)] = Rmfs[offset.y][offset.x][lid.y * 8 + lid.x];
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
















































// #include <metal_simdgroup_matrix>  // Available from Metal version 2.3 released with OS X 11.0+
// using namespace metal;
// 
// kernel void sgemm(device const float *data1, device const float *data2, device float *a, constant const uint64_t& N, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
//   constexpr const auto LID = 2;
//   a += gid.x * 32 * N + (gid.y * LID + lid.y) * 32;
//   data1 += gid.x * 32 * N;
//   data2 += (gid.y * LID + lid.y) * 32;
// 
//   simdgroup_float8x8 acc[4][4];
//   for (uint i = 0; i < 4; i++) {
//     for (uint j = 0; j < 4; j++) {
//       acc[i][j] = simdgroup_float8x8(0);
//     }
//   }
// 
//   simdgroup_float8x8 A[4];
//   simdgroup_float8x8 B[4];
//   for (uint k = 0; k < N; k+=8) {
//     threadgroup_barrier(mem_flags::mem_threadgroup);
//     simdgroup_load(A[0], data1+k+(0*N), N, ulong2(0, 0));
//     simdgroup_load(A[1], data1+k+(8*N), N, ulong2(0, 0));
//     simdgroup_load(A[2], data1+k+(16*N), N, ulong2(0, 0));
//     simdgroup_load(A[3], data1+k+(24*N), N, ulong2(0, 0));
//     simdgroup_load(B[0], data2+0+k*N, N, ulong2(0, 0));
//     simdgroup_load(B[1], data2+8+k*N, N, ulong2(0, 0));
//     simdgroup_load(B[2], data2+16+k*N, N, ulong2(0, 0));
//     simdgroup_load(B[3], data2+24+k*N, N, ulong2(0, 0));
// 
//     simdgroup_multiply_accumulate(acc[0][0], A[0], B[0], acc[0][0]);
//     simdgroup_multiply_accumulate(acc[0][1], A[1], B[0], acc[0][1]);
//     simdgroup_multiply_accumulate(acc[0][2], A[2], B[0], acc[0][2]);
//     simdgroup_multiply_accumulate(acc[0][3], A[3], B[0], acc[0][3]);
//     simdgroup_multiply_accumulate(acc[1][0], A[0], B[1], acc[1][0]);
//     simdgroup_multiply_accumulate(acc[1][1], A[1], B[1], acc[1][1]);
//     simdgroup_multiply_accumulate(acc[1][2], A[2], B[1], acc[1][2]);
//     simdgroup_multiply_accumulate(acc[1][3], A[3], B[1], acc[1][3]);
//     simdgroup_multiply_accumulate(acc[2][0], A[0], B[2], acc[2][0]);
//     simdgroup_multiply_accumulate(acc[2][1], A[1], B[2], acc[2][1]);
//     simdgroup_multiply_accumulate(acc[2][2], A[2], B[2], acc[2][2]);
//     simdgroup_multiply_accumulate(acc[2][3], A[3], B[2], acc[2][3]);
//     simdgroup_multiply_accumulate(acc[3][0], A[0], B[3], acc[3][0]);
//     simdgroup_multiply_accumulate(acc[3][1], A[1], B[3], acc[3][1]);
//     simdgroup_multiply_accumulate(acc[3][2], A[2], B[3], acc[3][2]);
//     simdgroup_multiply_accumulate(acc[3][3], A[3], B[3], acc[3][3]);
//   }
//   simdgroup_store(acc[0][0], a+(0+0*N), N, ulong2(0, 0));
//   simdgroup_store(acc[1][0], a+(8+0*N), N, ulong2(0, 0));
//   simdgroup_store(acc[2][0], a+(16+0*N), N, ulong2(0, 0));
//   simdgroup_store(acc[3][0], a+(24+0*N), N, ulong2(0, 0));
//   simdgroup_store(acc[0][1], a+(0+8*N), N, ulong2(0, 0));
//   simdgroup_store(acc[1][1], a+(8+8*N), N, ulong2(0, 0));
//   simdgroup_store(acc[2][1], a+(16+8*N), N, ulong2(0, 0));
//   simdgroup_store(acc[3][1], a+(24+8*N), N, ulong2(0, 0));
//   simdgroup_store(acc[0][2], a+(0+16*N), N, ulong2(0, 0));
//   simdgroup_store(acc[1][2], a+(8+16*N), N, ulong2(0, 0));
//   simdgroup_store(acc[2][2], a+(16+16*N), N, ulong2(0, 0));
//   simdgroup_store(acc[3][2], a+(24+16*N), N, ulong2(0, 0));
//   simdgroup_store(acc[0][3], a+(0+24*N), N, ulong2(0, 0));
//   simdgroup_store(acc[1][3], a+(8+24*N), N, ulong2(0, 0));
//   simdgroup_store(acc[2][3], a+(16+24*N), N, ulong2(0, 0));
//   simdgroup_store(acc[3][3], a+(24+24*N), N, ulong2(0, 0));
// }
