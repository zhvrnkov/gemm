#include <cstring>
#include <iostream>
#include <Foundation/Foundation.h>
#include <Accelerate/Accelerate.h>
#include <cstdio>
#include <simd/vector_types.h>
#include <vecLib/vDSP.h>
#include <simd/simd.h>
#include <time.h>

#define N 2048

alignas(16) float A[N * N];
alignas(16) float B[N * N];
alignas(16) float C[N * N];
alignas(16) float Bt[N * N];
float vdspResult[N * N];

void naive_gemm(float* A, float* B, float* C)
{
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        C[i * N + j] += A[i * N + k] * B[k * N + j];
      }
    }
  }
}

void gemm(float* A, float* B, float* C)
{
  constexpr auto block_x = 2;
  constexpr auto block_y = 4;
  constexpr auto tblock = 1;

  // transpose in blocks
  for (int y = 0; y < N; y += tblock) {
    for (int x = 0; x < N; x += tblock) {
      for (int yb = 0; yb < tblock; yb++) {
        for (int xb = 0; xb < tblock; xb++) {
          Bt[x * N + y + yb * N + xb] = B[y * N + x + yb * N + xb];
        }
      }
    }
  }

  for (int y = 0; y < N; y += block_y) {
    for (int x = 0; x < N; x += block_x) {
      float block_C[block_x * block_y] = {0};

      for (int k = 0; k < N; k++) {
        for (int yb = 0; yb < block_y; yb++) {
          for (int xb = 0; xb < block_x; xb++) {
            block_C[yb * block_y + xb] += A[y * N + yb * N + k] * Bt[x * N + xb * N + k];     
          }
        }
      }

      for (int i = 0; i < block_y * block_x; i++) {
        C[y * N + x + (i / block_y * N) + (i % block_x)] += block_C[i];
      }
    }
  }
}

int main()
{
  uint64_t flops = 2ul * N * N * N;
  printf("GFLOP %.3f\n", (float)flops * 1e-9);

  for (int i = 0; i < N; i++) {
    A[i] = (float)(rand() % 100);
    B[i] = (float)(rand() % 100);
  }

  for (int i = 0; i < 10; i++) {
    clock_t st = clock();

    vDSP_mmul(A, 1, B, 1, vdspResult, 1, N, N, N);

    clock_t et = clock();
    double tt = (double)(et - st) / CLOCKS_PER_SEC;

    printf("vdsp GFLOPS %.3f\n", (double)flops * 1e-9 / tt);
  }

  {
    clock_t st = clock();

    gemm(A, B, C);

    clock_t et = clock();
    double tt = (double)(et - st) / CLOCKS_PER_SEC;

    printf("gemm GFLOPS %.3f\n", (double)flops * 1e-9 / tt);

    for (uint i = 0; i < N * N; i++) {
      assert(C[i] == vdspResult[i]);
    }
  }

  {
    memset(C, 0, sizeof(C));
    clock_t st = clock();

    naive_gemm(A, B, C);

    clock_t et = clock();
    double tt = (double)(et - st) / CLOCKS_PER_SEC;

    printf("naive gemm GFLOPS %.3f\n", (double)flops * 1e-9 / tt);
  }
  

  return 0;
}

