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
        auto kernelFunc = [gpu::lib newFunctionWithName:@"sgemm_32x32"];
        kernel = [gpu::device newComputePipelineStateWithFunction:kernelFunc error:nil];
    });
    if (!kernel) {
        NSLog(@"got error during pipeline creation");
        return;
    }
    
    constexpr auto dim = 4;
    uint64_t N = A.rows;
    MTLSize tgroupSize;
    tgroupSize.width = 32;
    tgroupSize.height = 2;
    tgroupSize.depth = 1;
    auto encoder = [cmd computeCommandEncoder];
    
    [encoder setBuffer:A.data offset:0 atIndex:0];
    [encoder setBuffer:B.data offset:0 atIndex:1];
    [encoder setBuffer:C.data offset:0 atIndex:2];
    [encoder setBytes:(void*)&N length:sizeof(N) atIndex:3];
    //    [encoder setThreadgroupMemoryLength:(32 * 32 * sizeof(float)) atIndex:0];
    //    [encoder setThreadgroupMemoryLength:(32 * 32 * sizeof(float)) atIndex:1];
    [encoder setComputePipelineState:kernel];
    [encoder dispatchThreadgroups:MTLSizeMake(A.columns / (dim * 8), A.rows / (dim * 8 * 2), 1) threadsPerThreadgroup:tgroupSize];
    // [encoder dispatchThreadgroups:MTLSizeMake(N / (8 * 4), N / (8 * 4 * 2), 1) threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];
    
    [encoder endEncoding];
}
}
namespace gpugemv {
void encode(id<MTLCommandBuffer> cmd, MPSMatrix* mat, MPSVector* vec, MPSVector* output)
{
    assert(mat.columns == vec.length);
    assert(mat.rows == output.length);
    
    static id<MTLComputePipelineState> kernel;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        auto kernelFunc = [gpu::lib newFunctionWithName:@"sgemv"];
        kernel = [gpu::device newComputePipelineStateWithFunction:kernelFunc error:nil];
    });
    if (!kernel) {
        NSLog(@"got error during pipeline creation");
        return;
    }
    
    MTLSize tgroupSize;
    tgroupSize.width = 32;
    tgroupSize.height = 2;
    tgroupSize.depth = 1;

    uint32_t H = (uint32_t)mat.rows;
    uint32_t W = (uint32_t)mat.columns;
    auto encoder = [cmd computeCommandEncoder];
    [encoder setBuffer:mat.data offset:0 atIndex:0];
    [encoder setBuffer:vec.data offset:0 atIndex:1];
    [encoder setBuffer:output.data offset:0 atIndex:2];
    [encoder setBytes:(void*)&H length:sizeof(H) atIndex:3];
    [encoder setBytes:(void*)&W length:sizeof(W) atIndex:4];
//    [encoder setThreadgroupMemoryLength:tgroupSize.width * sizeof(float) atIndex:0];
    [encoder setComputePipelineState:kernel];
    
    [encoder dispatchThreadgroups:MTLSizeMake(1, H / tgroupSize.height, 1) threadsPerThreadgroup:tgroupSize];
    [encoder endEncoding];
}
}

constexpr auto N = 1024 * 8;
constexpr auto M = 16384 / 1;

int main_vec()
{
    auto* mat = new std::array<float, N * M>{};
    auto* vec = new std::array<float, M>{};
    auto* vdspOutVec = new std::array<float, N>{};
    
    std::normal_distribution<float> dstr(0.0, 5.0);
    std::mt19937 gen{};
    
    for (int i = 0; i < N * M; i++) {
        mat->at(i) = dstr(gen);
        if (i < vec->size()) {
            vec->at(i) = dstr(gen);
        }
    }
    vDSP_mmul(mat->data(), 1, vec->data(), 1, vdspOutVec->data(), 1, N, 1, M);

    auto buffMat = [gpu::device newBufferWithBytes:mat->data() length:mat->size() * sizeof(float) options:MTLResourceStorageModeShared];
    auto buffVec = [gpu::device newBufferWithBytes:vec->data() length:vec->size() * sizeof(float) options:MTLResourceStorageModeShared];
    auto buffMpsOutVec = [gpu::device newBufferWithLength:vdspOutVec->size() * sizeof(float) options:MTLResourceStorageModeShared];
    auto buffOutVec = [gpu::device newBufferWithLength:vdspOutVec->size() * sizeof(float) options:MTLResourceStorageModeShared];

    uint64_t flops = 2ul * N * M;
    printf("GFLOP %.3f\n", (float)flops * 1e-9);
    
    auto matDescriptor = [MPSMatrixDescriptor matrixDescriptorWithRows:N columns:M rowBytes:M * sizeof(float) dataType:MPSDataTypeFloat32];
    auto mpsMat = [[MPSMatrix alloc] initWithBuffer:buffMat descriptor:matDescriptor];
    
    auto vecDescriptor = [MPSVectorDescriptor vectorDescriptorWithLength:M dataType:MPSDataTypeFloat32];
    auto outVecDescriptor = [MPSVectorDescriptor vectorDescriptorWithLength:N dataType:MPSDataTypeFloat32];
    auto mpsVec = [[MPSVector alloc] initWithBuffer:buffVec descriptor:vecDescriptor];
    auto mpsOutVec = [[MPSVector alloc] initWithBuffer:buffMpsOutVec descriptor:outVecDescriptor];
    auto outVec = [[MPSVector alloc] initWithBuffer:buffOutVec descriptor:outVecDescriptor];
    
    auto kernel = [[MPSMatrixVectorMultiplication alloc] initWithDevice:gpu::device transpose:NO rows:N columns:M alpha:1.0 beta:1.0];
    
    constexpr auto ITERS = 16;
    double mpsTotalTime = 0.0;
    for (int i = 0; i < ITERS; i++) {
//    while (true) {
        memset(buffMpsOutVec.contents, 0, buffMpsOutVec.length);
        auto cmd = [gpu::queue commandBuffer];
        [kernel encodeToCommandBuffer:cmd inputMatrix:mpsMat inputVector:mpsVec resultVector:mpsOutVec];
        [cmd commit];
        [cmd waitUntilCompleted];
        auto cmdTime = [cmd GPUEndTime] - [cmd GPUStartTime];
        mpsTotalTime += cmdTime;
    }
    
    double gemvTotalTime = 0.0;
    for (int i = 0; i < ITERS; i++) {
//    while (true) {
        memset(buffOutVec.contents, 0, buffOutVec.length);
        auto cmd = [gpu::queue commandBuffer];
        gpugemv::encode(cmd, mpsMat, mpsVec, outVec);
        [cmd commit];
        [cmd waitUntilCompleted];
        auto cmdTime = [cmd GPUEndTime] - [cmd GPUStartTime];
        gemvTotalTime += cmdTime;
    }
    
    printf("MPS:   AVG %.3f GFLOP/s\n", (double)flops / (mpsTotalTime / (double)ITERS) * 1e-9);
    printf("SGEMV: AVG %.3f GFLOP/s\n", (double)flops / (gemvTotalTime / (double)ITERS) * 1e-9);
    float epsilon = 0.01;
    for (auto i = 0; i < N; i++) {
        auto sgemvX = ((float*)buffOutVec.contents)[i];
        auto mpsX = ((float*)buffMpsOutVec.contents)[i];
        auto vdspX = vdspOutVec->at(i);
        if (fabsf(vdspX - mpsX) > epsilon) {
            printf("missmatch at %d (%f != %f)\n", i, vdspX, mpsX);
            assert(false);
        }
        if (fabsf(vdspX - sgemvX) > epsilon) {
            printf("missmatch at %d (%f != %f) %f\n", i, vdspX, sgemvX, fabsf(vdspX - sgemvX));
            assert(false);
        }
    }
    
    return 0;
}

int main_mat()
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
    
//    for (int i = 0; i < 3; i++) {
    while (true) {
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
    
    // while (true) {
    for (int i = 0; i < 3; i++) {
        memset(buffC.contents, 0, buffC.length);
        auto cmd = [gpu::queue commandBuffer];
        gpugemm::encode(cmd, matA, matB, matC);
        [cmd commit];
        [cmd waitUntilCompleted];
        auto cmdTime = [cmd GPUEndTime] - [cmd GPUStartTime];
        printf("SGEMM: %.3f TFLOP/s\n", (double)flops / cmdTime * 1e-12);
        
        if (false) {
            int blockX = 0;
            int blockY = 0;
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
    return 0;
}

int main_test()
{
    id<MTLComputePipelineState> kernel;
    auto kernelFunc = [gpu::lib newFunctionWithName:@"simd_test"];
    kernel = [gpu::device newComputePipelineStateWithFunction:kernelFunc error:nil];
    if (!kernel) {
        NSLog(@"got error during pipeline creation");
        return -1;
    }
    
    constexpr auto N = 64;
    auto buffA = [gpu::device newBufferWithLength:N * sizeof(float) options:MTLResourceStorageModeShared];
    auto buffB = [gpu::device newBufferWithLength:N * sizeof(float) options:MTLResourceStorageModeShared];
    auto buffC = [gpu::device newBufferWithLength:N * sizeof(float) options:MTLResourceStorageModeShared];
    auto vdspC = new float[N];
    for (int i = 0; i < (buffA.length / sizeof(float)); i++) {
        ((float*)buffA.contents)[i] = rand() % 128;
        ((float*)buffB.contents)[i] = rand() % 128;
        ((float*)buffC.contents)[i] = 0;
    }
    auto cmd = [gpu::queue commandBuffer];
    
    auto encoder = [cmd computeCommandEncoder];
    [encoder setBuffer:buffA offset:0 atIndex:0];
    [encoder setBuffer:buffB offset:0 atIndex:1];
    [encoder setBuffer:buffC offset:0 atIndex:2];
    [encoder setComputePipelineState:kernel];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [encoder endEncoding];
    
    [cmd commit];
    [cmd waitUntilCompleted];
    
    const auto sN = (int)std::sqrt(N);
    vDSP_mmul((float*)buffA.contents, 1, (float*)buffB.contents, 1, vdspC, 1, sN, sN, sN);
    
    for (int y = 0; y < sN; y++) {
        for (int x = 0; x < sN; x++) {
            printf("%6.2f ", ((float*)buffC.contents)[y * sN + x]);
        }
        printf("\n");
    }
    printf("\n");
    printf("\n");
    for (int y = 0; y < sN; y++) {
        for (int x = 0; x < sN; x++) {
            printf("%6.2f ", vdspC[y * sN + x]);
        }
        printf("\n");
    }
    printf("\n");
    return 0;
}

int main()
{
    return main_vec();
}
