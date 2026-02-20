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

#define N 512

alignas(32) float A[N * N];
alignas(32) float B[N * N];
alignas(32) float C[N * N];
alignas(32) float Bt[N * N];
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

template<int length, int block>
void transpose(float* m, float* mT)
{
  // transpose in blocks
  for (int y = 0; y < length; y += block) {
    for (int x = 0; x < length; x += block) {
      for (int yb = 0; yb < block; yb++) {
        for (int xb = 0; xb < block; xb++) {
          mT[y + x * length + yb * length + xb] = m[y * length + x + yb * length + xb];
        }
      }
    }
  }
}

constexpr auto block_x = 4;
constexpr auto block_y = 4;
void gemm(float* A, float* B, float* C, int sy, int ey)
{
  for (int y = sy; y < ey; y += block_y) {
    for (int x = 0; x < N; x += block_x) {
      // y and x is the block
      // to compute the whole block we need to go through block_y A rows and block_x B cols

      // block_C is in cache since its relatively small and local
      float block_C[block_x * block_y] = {0};
      float* block_A = &A[y * N];
      float* block_B = &Bt[x * N];

      // after this loop (kernel) block_C is complete
      // iterating over ks, this is outer loop, so block_C contain partial dot products
      for (int k = 0; k < N; k += 1) {
        for (int yb = 0; yb < block_y; yb++) {
          // A[y * N + yb * N + k]... is in cache now, but yb is strided by N.
          // is that a problem for L1 cache?
          // if block_y is 4, then 32KB of A is used for this K loop
          float tA = block_A[yb * N + k];
          for (int xb = 0; xb < block_x; xb++) {
            // B[x * N + xb * N + k] we go through xb with stride N
            // so whole B for 0<=xb<4 is in cache?
            block_C[yb * block_x + xb] += tA * block_B[xb * N + k];     
          }
        }
      }

      for (int yb = 0; yb < block_y; yb++) {
        for (int xb = 0; xb < block_x; xb++) {
          C[y * N + yb * N + x + xb] = block_C[yb * block_x + xb];
        }
      }
    }
  }
}

constexpr auto nthreads = 8;
std::mutex mtx{};
std::condition_variable cv{};
bool dataReady = false;

void gemm_thread(int threadIdx) {
  std::unique_lock<std::mutex> lck{mtx};
  cv.wait(lck, [] { return dataReady; });

  constexpr auto totalBlocks = N / block_y;
  const auto blocksPerThread = totalBlocks / nthreads;
  gemm(A, B, C, threadIdx * blocksPerThread * block_y, (threadIdx + 1) * blocksPerThread * block_y);
}

int main()
{
  uint64_t flops = 2ul * N * N * N;
  printf("GFLOP %.3f\n", (float)flops * 1e-9);

  for (int i = 0; i < N * N; i++) {
    A[i] = (float)(rand() % 100);
    B[i] = (float)(rand() % 100);
  }

  for (int i = 0; i < 1; i++) {
    clock_t st = clock();

    vDSP_mmul(A, 1, B, 1, vdspResult, 1, N, N, N);

    clock_t et = clock();
    double tt = (double)(et - st) / CLOCKS_PER_SEC;

    printf("vdsp GFLOPS %.3f\n", (double)flops * 1e-9 / tt);
  }

  for (int i = 0; i < 10; i++) {
    memset(C, 0, sizeof(C));
    transpose<N, 1>(B, Bt);
    clock_t st = clock();

    gemm(A, B, C, 0, N);

    clock_t et = clock();
    double tt = (double)(et - st) / CLOCKS_PER_SEC;

    printf("gemm GFLOPS %.3f\n", (double)flops * 1e-9 / tt);

    for (uint i = 0; i < N * N; i++) {
      if (C[i] != vdspResult[i]) {
        printf("missmatch at %d : %f != %f\n", i, C[i], vdspResult[i]);
        assert(false);
      }
    }
  }

  return 0;

  for (int i = 0; i < 0; i++) {
    memset(C, 0, sizeof(C));
    std::vector<std::thread> threads;

    for (int i = 0; i < nthreads; i++) {
      threads.emplace_back(gemm_thread, i);
    }


    clock_t st = clock();
    // transpose(B, Bt);
    {
      std::lock_guard<std::mutex> lock{mtx};
      dataReady = true;
    }
    cv.notify_all();


    for (auto& t : threads) {
      t.join();
    }

    clock_t et = clock();
    double tt = (double)(et - st) / CLOCKS_PER_SEC;

    printf("multi gemm GFLOPS %.3f\n", (double)flops * 1e-9 / tt);

    for (uint i = 0; i < N * N; i++) {
      if (C[i] != vdspResult[i]) {
        printf("missmatch at %d : %f != %f\n", i, C[i], vdspResult[i]);
        assert(false);
      }
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
  transpose<L, 2>(mat, matT);
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

