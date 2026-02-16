#include <cstring>
#include <iostream>
#include <Foundation/Foundation.h>
#include <Accelerate/Accelerate.h>
#include <cstdio>
#include <vecLib/vDSP.h>
#include <time.h>

#define N 2048

float A[N * N];
float B[N * N];
float C[N * N];
float Bt[N * N];
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
  constexpr auto block = 16;

  for (int y = 0; y < N; y += block) {
    for (int x = 0; x < N; x += block) {
      for (int k = 0; k < N; k += block) {
        float* block_A = A + y * N + k;
        float* block_B = B + k * N + x;
        float block_C[block * block] = {0};

        for (int yb = 0; yb < block; yb++) {
          for (int xb = 0; xb < block; xb++) {
            for (int kb = 0; kb < block; kb++) {
              block_C[yb * block + xb] += block_A[yb * N + kb] * block_B[kb * N + xb];     
            }
          }
        }
        for (int i = 0; i < block * block; i++) {
          C[y * N + x + (i / block * N) + (i % block)] += block_C[i];
        }
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
    constexpr auto tblock = 8;
    for (int y = 0; y < N; y += tblock) {
      for (int x = 0; x < N; x += tblock) {
        for (int yb = 0; yb < tblock; yb++) {
          for (int xb = 0; xb < tblock; xb++) {
            Bt[x * N + y + xb * N + yb] = B[y * N + x + yb * N + xb];
          }
        }
      }
    }

    clock_t st = clock();

    gemm(A, B, C);

    clock_t et = clock();
    double tt = (double)(et - st) / CLOCKS_PER_SEC;

    printf("gemm GFLOPS %.3f\n", (double)flops * 1e-9 / tt);
    memset(C, 0, sizeof(C));
  }

  {
    clock_t st = clock();

    naive_gemm(A, B, C);

    clock_t et = clock();
    double tt = (double)(et - st) / CLOCKS_PER_SEC;

    printf("naive gemm GFLOPS %.3f\n", (double)flops * 1e-9 / tt);
  }

  for (uint i = 0; i < N * N; i++) {
    assert(C[i] == vdspResult[i]);
  }
  

  return 0;
}

