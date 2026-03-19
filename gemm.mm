// clang++ -ffast-math -O3 -std=c++20 gemm.mm -framework Accelerate

#include <condition_variable>
#include <cstring>
#include <iostream>
#include <cstdio>
#include <cassert>
#include <mutex>
#include <simd/vector_types.h>
#include <vecLib/vDSP.h>
#include <simd/simd.h>
#include <time.h>
#include <thread>
#include <shared_mutex>
#include <condition_variable>

// mat of size (rows x cols)
// A of size (M x N)
// B of size (N x P)
// C of size (M x P)
void naive_gemm(const float* A, const float* B, float* C, int64_t M, int64_t N, int64_t P)
{
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < P; j++) {
      for (int k = 0; k < N; k++) {
        C[i * P + j] += A[i * N + k] * B[k * P + j];
      }
    }
  }
}

  template<int BLOCK>
void transpose(const float* m, float* mT, int64_t M, int64_t N)
{
  // transpose in blocks
  for (int y = 0; y < M; y += BLOCK) {
    for (int x = 0; x < N; x += BLOCK) {
      for (int yb = 0; yb < BLOCK; yb++) {
        for (int xb = 0; xb < BLOCK; xb++) {
          mT[y + x * M + yb * M + xb] = m[y * N + x + yb * N + xb];
        }
      }
    }
  }
}

  template<int64_t BLOCK_X, int64_t BLOCK_Y>
void gemm(const float* A, const float* B, float* C, 
    int64_t M, int64_t N, int64_t P, 
    bool transposeB)
{
  const float* Bt;
  std::unique_ptr<const float> Btptr;
  if (transposeB) {
    float* tmpBt = new float[N * P];
    transpose<1>(B, tmpBt, N, P);
    Btptr = std::unique_ptr<const float>(tmpBt);
    Bt = Btptr.get();
  } else {
    Bt = B;
  }

  for (int y = 0; y < M; y += BLOCK_Y) {
    for (int x = 0; x < P; x += BLOCK_X) {
      // y and x is the block
      // to compute the whole block we need to go through BLOCK_Y A rows and BLOCK_X B cols

      // block_C is in cache since its relatively small and local
      float block_C[BLOCK_X * BLOCK_Y] = {0};
      const float* block_A = &A[y * N];
      const float* block_B = &Bt[x * N];

      // after this loop (kernel) block_C is complete
      // iterating over ks, this is outer loop, so block_C contain partial dot products
      for (int k = 0; k < N; k += 1) {
        for (int yb = 0; yb < BLOCK_Y; yb++) {
          // A[y * N + yb * N + k]... is in cache now, but yb is strided by N.
          // is that a problem for L1 cache?
          // if BLOCK_Y is 4, then 32KB of A is used for this K loop
          float tA = block_A[yb * N + k];
          for (int xb = 0; xb < BLOCK_X; xb++) {
            // B[x * N + xb * N + k] we go through xb with stride N
            // so whole B for 0<=xb<4 is in cache?
            block_C[yb * BLOCK_X + xb] += tA * block_B[xb * N + k];     
          }
        }
      }

      for (int yb = 0; yb < BLOCK_Y; yb++) {
        for (int xb = 0; xb < BLOCK_X; xb++) {
          C[y * P + yb * P + x + xb] = block_C[yb * BLOCK_X + xb];
        }
      }
    }
  }
}

constexpr auto M = 2048;
constexpr auto N = 2048;
constexpr auto P = 2048;
alignas(32) float A[M * N];
alignas(32) float B[N * P];
alignas(32) float C[M * P];
alignas(32) float Bt[P * N];
float vdspResult[M * P];

int main()
{

  uint64_t flops = 2ul * M * N * P;
  printf("GFLOP %.3f\n", (float)flops * 1e-9);

  for (int i = 0; i < M * N; i++) {
    A[i] = (float)(rand() % 100);
  }
  for (int i = 0; i < N * P; i++) {
    B[i] = (float)(rand() % 100);
  }

  for (int i = 0; i < 1; i++) {
    clock_t st = clock();

    vDSP_mmul(A, 1, B, 1, vdspResult, 1, M, P, N);

    clock_t et = clock();
    double tt = (double)(et - st) / CLOCKS_PER_SEC;

    printf("vdsp GFLOPS %.3f\n", (double)flops * 1e-9 / tt);
  }

  constexpr auto ITERS = 10;
  double gemmTotalTime = 0.0;
  for (int i = 0; i < ITERS; i++) {
    memset(C, 0, sizeof(C));
    transpose<1>(B, Bt, N, P);
    clock_t st = clock();

    gemm<4, 4>(A, Bt, C, M, N, P, false);

    clock_t et = clock();
    double tt = (double)(et - st) / CLOCKS_PER_SEC;
    gemmTotalTime += tt;

    for (uint i = 0; i < M * P; i++) {
      if (C[i] != vdspResult[i]) {
        printf("missmatch at %d : %f != %f\n", i, C[i], vdspResult[i]);
        assert(false);
      }
    }
  }

  printf("gemm AVG GFLOPS %.3f\n", (double)flops * 1e-9 / (gemmTotalTime / (double)ITERS));

  return 0;
}

int main_debug()
{
  constexpr auto L = 4;
  float mat[L * L] = {0};
  for (int i = 0; i < L * L; i++) mat[i] = i;

  for (int y = 0; y < L; y++) {
    for (int x = 0; x < L; x++) {
      printf("%5.2f ", mat[y * L + x]);
    }
    printf("\n");
  }
  printf("\n\n");

  float matT[L * L] = {0};
  transpose<2>(mat, matT, L, L);
  for (int y = 0; y < L; y++) {
    for (int x = 0; x < L; x++) {
      printf("%5.2f ", matT[y * L + x]);
    }
    printf("\n");
  }
  printf("\n");

  printf("DEBUG\n");
  return 0;
}

