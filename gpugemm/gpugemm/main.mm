#include <cassert>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <iostream>
#import <Foundation/Foundation.h>
#include <objc/objc.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <vecLib/vDSP.h>

#include <vector>
#include <span>
#include <fstream>
#include <random>

namespace gpu {
auto device = MTLCreateSystemDefaultDevice();
auto queue = [device newCommandQueue];
auto lib = [device newDefaultLibrary];

namespace compute {
void dispatch1d(id<MTLComputeCommandEncoder> encoder,
                id<MTLComputePipelineState> kernel,
                uint64_t size)
{
    uint64_t simdgroupSize = kernel.threadExecutionWidth;
    uint64_t simdgroupsInSize = (size + simdgroupSize - 1) / simdgroupSize;
    auto threadsPerThreadgroup = std::min(simdgroupSize * simdgroupsInSize, (uint64_t)encoder.device.maxThreadsPerThreadgroup.width);
    [encoder setComputePipelineState:kernel];
    [encoder dispatchThreads:MTLSizeMake(size, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadsPerThreadgroup, 1, 1)];
}
}
}

namespace gpugemm {
void encode(id<MTLCommandBuffer> cmd, MPSMatrix* A, MPSMatrix* B, MPSMatrix* C)
{
    assert(A.rows == A.columns);
    assert(B.rows == B.columns);
    assert(C.rows == C.columns);
    assert(A.rows == B.columns);
    assert(B.rows == C.columns);

    static id<MTLComputePipelineState> kernel;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        auto kernelFunc = [gpu::lib newFunctionWithName:@"sgemm"];
        kernel = [gpu::device newComputePipelineStateWithFunction:kernelFunc error:nil];
    });
    if (!kernel) {
        NSLog(@"got error during pipeline creation");
        return;
    }

    uint64_t N = A.rows;
    MTLSize tgroupSize;
    tgroupSize.width = 32;
    tgroupSize.height = 32;
    tgroupSize.depth = 1;
    auto encoder = [cmd computeCommandEncoder];

    [encoder setBuffer:A.data offset:0 atIndex:0];
    [encoder setBuffer:B.data offset:0 atIndex:1];
    [encoder setBuffer:C.data offset:0 atIndex:2];
    [encoder setBytes:(void*)&N length:sizeof(N) atIndex:3];
//    [encoder setThreadgroupMemoryLength:(32 * 32 * sizeof(float)) atIndex:0];
//    [encoder setThreadgroupMemoryLength:(32 * 32 * sizeof(float)) atIndex:1];
    [encoder setComputePipelineState:kernel];
    [encoder dispatchThreadgroups:MTLSizeMake(A.columns / tgroupSize.width, A.rows / tgroupSize.height, 1) threadsPerThreadgroup:tgroupSize];
    // [encoder dispatchThreadgroups:MTLSizeMake(N / (8 * 4), N / (8 * 4 * 2), 1) threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];

    [encoder endEncoding];
}
}

constexpr auto N = 4096;

int main()
{
  auto* A = new std::array<float, N * N>{};
  auto* B = new std::array<float, N * N>{};
  auto* C = new std::array<float, N * N>{};
  auto* vdspC = new std::array<float, N * N>{};

  for (int i = 0; i < N * N; i++) {
    A->at(i) = (float)(rand() % 100);
    B->at(i) = (float)(rand() % 100);
    C->at(i) = 0;
    vdspC->at(i) = 0;
  }

  vDSP_mmul(A->data(), 1, B->data(), 1, vdspC->data(), 1, N, N, N);

  auto buffA = [gpu::device newBufferWithLength:N * N * sizeof(float) options:MTLResourceStorageModeShared];
  auto buffB = [gpu::device newBufferWithLength:N * N * sizeof(float) options:MTLResourceStorageModeShared];
  auto mpsBuffC = [gpu::device newBufferWithLength:N * N * sizeof(float) options:MTLResourceStorageModeShared];
  auto buffC = [gpu::device newBufferWithLength:N * N * sizeof(float) options:MTLResourceStorageModeShared];

  for (int i = 0; i < N * N; i++) {
    ((float*)buffA.contents)[i] = A->at(i);
    ((float*)buffB.contents)[i] = B->at(i);
    ((float*)mpsBuffC.contents)[i] = C->at(i);
  }

  uint64_t flops = 2ul * N * N * N;
  printf("TFLOP %.3f\n", (float)flops * 1e-12);

  auto matDescriptor = [MPSMatrixDescriptor matrixDescriptorWithRows:N columns:N rowBytes:N * sizeof(float) dataType:MPSDataTypeFloat32];
  auto matA = [[MPSMatrix alloc] initWithBuffer:buffA descriptor:matDescriptor];
  auto matB = [[MPSMatrix alloc] initWithBuffer:buffB descriptor:matDescriptor];
  auto mpsMatC = [[MPSMatrix alloc] initWithBuffer:mpsBuffC descriptor:matDescriptor];
  auto matC = [[MPSMatrix alloc] initWithBuffer:buffC descriptor:matDescriptor];

  auto kernel = [[MPSMatrixMultiplication alloc] initWithDevice:gpu::device transposeLeft:NO transposeRight:NO resultRows:N resultColumns:N interiorColumns:N alpha:1.0 beta:1.0];

  for (int i = 0; i < 1; i++) {
//    while (true) {
    memset(mpsBuffC.contents, 0, mpsBuffC.length);
    auto cmd = [gpu::queue commandBuffer];
    [kernel encodeToCommandBuffer:cmd leftMatrix:matA rightMatrix:matB resultMatrix:mpsMatC];
    [cmd commit];
    [cmd waitUntilCompleted];
    auto cmdTime = [cmd GPUEndTime] - [cmd GPUStartTime];
    printf("MPS:   %.3f TFLOP/s\n", (double)flops / cmdTime * 1e-12);

    for (auto i = 0; i < vdspC->size(); i++)
      assert(vdspC->at(i) == ((float*)mpsBuffC.contents)[i]);
  }

  for (int i = 0; i < 4; i++) {
    memset(buffC.contents, 0, buffC.length);
    auto cmd = [gpu::queue commandBuffer];
    gpugemm::encode(cmd, matA, matB, matC);
    [cmd commit];
    [cmd waitUntilCompleted];
    auto cmdTime = [cmd GPUEndTime] - [cmd GPUStartTime];
    printf("SGEMM: %.3f TFLOP/s\n", (double)flops / cmdTime * 1e-12);

    if (false) {
      int blockX = 2;
      int blockY = 2;
      float* block = ((float*)buffC.contents) + blockY * 32 * N + blockX * 32;
      for (int by = 0; by < 32; by++) {
        for (int bx = 0; bx < 32; bx++) {
          printf("%5.1f ", block[by * N + bx]);
        }
        printf("\n");
      }
      printf("\n");
    }

    for (auto i = 0; i < vdspC->size(); i++) {
      auto x = ((float*)buffC.contents)[i];
      if (vdspC->at(i) != x) {
        printf("missmatch at %d (%f != %f)\n", i, vdspC->at(i), x);
        assert(false);
      }
    }
  }

}
