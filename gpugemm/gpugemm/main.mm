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
    assert(A.columns == B.rows);
    assert(A.rows == C.rows);
    assert(B.columns == C.columns);
   
    static id<MTLComputePipelineState> kernel;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        auto kernelFunc = [gpu::lib newFunctionWithName:@"sgemm_32x32_unrolled"];
        kernel = [gpu::device newComputePipelineStateWithFunction:kernelFunc error:nil];
    });
    if (!kernel) {
        NSLog(@"got error during pipeline creation");
        return;
    }
    
    constexpr auto dim = 4;
    uint64_t M = A.rows;
    uint64_t N = A.columns;
    uint64_t P = C.columns;
    MTLSize tgroupSize;
    tgroupSize.width = 32;
    tgroupSize.height = 2;
    tgroupSize.depth = 1;
    auto encoder = [cmd computeCommandEncoder];
    
    [encoder setBuffer:A.data offset:0 atIndex:0];
    [encoder setBuffer:B.data offset:0 atIndex:1];
    [encoder setBuffer:C.data offset:0 atIndex:2];
    [encoder setBytes:(void*)&M length:sizeof(M) atIndex:3];
    [encoder setBytes:(void*)&N length:sizeof(N) atIndex:4];
    [encoder setBytes:(void*)&P length:sizeof(P) atIndex:5];
    //    [encoder setThreadgroupMemoryLength:(32 * 32 * sizeof(float)) atIndex:0];
    //    [encoder setThreadgroupMemoryLength:(32 * 32 * sizeof(float)) atIndex:1];
    [encoder setComputePipelineState:kernel];
    [encoder dispatchThreadgroups:MTLSizeMake(P / (dim * 8), M / (dim * 8 * 2), 1) threadsPerThreadgroup:tgroupSize];
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

namespace gpudot {
void encode(id<MTLCommandBuffer> cmd, id<MTLBuffer> x, id<MTLBuffer> y, id<MTLBuffer> output)
{
    assert(x.length == y.length);
    const uint64_t N = x.length / sizeof(float);
    
    static id<MTLComputePipelineState> kernel;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        auto kernelFunc = [gpu::lib newFunctionWithName:@"dotpr"];
        kernel = [gpu::device newComputePipelineStateWithFunction:kernelFunc error:nil];
    });
    if (!kernel) {
        NSLog(@"got error during pipeline creation");
        return;
    }
    
    auto encoder = [cmd computeCommandEncoder];
    [encoder setBuffer:x offset:0 atIndex:0];
    [encoder setBuffer:y offset:0 atIndex:1];
    [encoder setBuffer:output offset:0 atIndex:2];
    [encoder setBytes:(void*)&N length:sizeof(N) atIndex:3];
    [encoder setComputePipelineState:kernel];
    
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
    [encoder endEncoding];
}

void encode2(id<MTLCommandBuffer> cmd, id<MTLBuffer> x, id<MTLBuffer> y, id<MTLBuffer> output)
{
    assert(x.length == y.length);
    const uint64_t N = x.length / sizeof(float);
    
    static id<MTLComputePipelineState> kernel0;
    static id<MTLComputePipelineState> kernel1;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        auto kernel0Func = [gpu::lib newFunctionWithName:@"dot_reduce0"];
        auto kernel1Func = [gpu::lib newFunctionWithName:@"dot_reduce1"];
        kernel0 = [gpu::device newComputePipelineStateWithFunction:kernel0Func error:nil];
        kernel1 = [gpu::device newComputePipelineStateWithFunction:kernel1Func error:nil];
    });
    if (!kernel0 || !kernel1) {
        NSLog(@"got error during pipeline creation");
        return;
    }
    
    auto threadsPerThreadgroup = MTLSizeMake(1024, 1, 1);
    auto threadgroupMemFloats = 1024 * 2;
    auto floatsPerThreadgroup = threadgroupMemFloats * 4;
    auto threadgroupsWidth = N / floatsPerThreadgroup;
    auto threadgroups = MTLSizeMake(threadgroupsWidth, 1, 1);
    id<MTLBuffer> interm = [gpu::device newBufferWithLength:threadgroups.width * sizeof(float) options:MTLResourceStorageModePrivate];
    
    auto encoder = [cmd computeCommandEncoder];
    [encoder setBuffer:x offset:0 atIndex:0];
    [encoder setBuffer:y offset:0 atIndex:1];
    [encoder setBuffer:interm offset:0 atIndex:2];
    [encoder setBytes:(void*)&N length:sizeof(N) atIndex:3];
    [encoder setThreadgroupMemoryLength:threadgroupMemFloats * sizeof(float) atIndex:0];
    [encoder setComputePipelineState:kernel0];
    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerThreadgroup];
    
    [encoder setBuffer:interm offset:0 atIndex:0];
    [encoder setBuffer:output offset:0 atIndex:1];
    [encoder setBytes:(void*)&threadgroupsWidth length:sizeof(threadgroupsWidth) atIndex:2];
    [encoder setThreadgroupMemoryLength:threadgroupsWidth * sizeof(float) atIndex:0];
    [encoder setComputePipelineState:kernel1];
    [encoder dispatchThreads:MTLSizeMake(threadgroupsWidth, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadgroupsWidth, 1, 1)];
    
    [encoder endEncoding];
}
}


int main_vec()
{
    constexpr auto N = 1024 * 8;
    constexpr auto M = 16384 / 1;
    
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
    constexpr auto M = 4096;
    constexpr auto N = 4096;
    constexpr auto P = 4096;
    
    std::normal_distribution<float> dstr(0.0, 5.0);
    std::mt19937 gen{};
    
    auto* A = new std::array<float, M * N>{};
    auto* B = new std::array<float, N * P>{};
    auto* C = new std::array<float, M * P>{};
    auto* vdspC = new std::array<float, M * P>{};
    
    auto maxSize = std::max({A->size(), B->size(), C->size()});
    for (int i = 0; i < maxSize; i++) {
        if (i < A->size()) A->at(i) = dstr(gen);
        if (i < B->size()) B->at(i) = dstr(gen);
        if (i < C->size()) C->at(i) = dstr(gen);
        if (i < vdspC->size()) vdspC->at(i) = dstr(gen);
    }
    
    vDSP_mmul(A->data(), 1, B->data(), 1, vdspC->data(), 1, M, P, N);
    
    auto buffA = [gpu::device newBufferWithBytes:A->data() length:A->size() * sizeof(float) options:MTLResourceStorageModeShared];
    auto buffB = [gpu::device newBufferWithBytes:B->data() length:B->size() * sizeof(float) options:MTLResourceStorageModeShared];
    auto mpsBuffC = [gpu::device newBufferWithBytes:C->data() length:C->size() * sizeof(float) options:MTLResourceStorageModeShared];
    auto buffC = [gpu::device newBufferWithBytes:C->data() length:C->size() * sizeof(float) options:MTLResourceStorageModeShared];
    
    uint64_t flops = 2ul * N * M * P;
    printf("TFLOP %.3f\n", (float)flops * 1e-12);
    
    auto matDescriptorA = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:N rowBytes:N * sizeof(float) dataType:MPSDataTypeFloat32];
    auto matDescriptorB = [MPSMatrixDescriptor matrixDescriptorWithRows:N columns:P rowBytes:P * sizeof(float) dataType:MPSDataTypeFloat32];
    auto matDescriptorC = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:P rowBytes:P * sizeof(float) dataType:MPSDataTypeFloat32];
    auto matA = [[MPSMatrix alloc] initWithBuffer:buffA descriptor:matDescriptorA];
    auto matB = [[MPSMatrix alloc] initWithBuffer:buffB descriptor:matDescriptorB];
    auto mpsMatC = [[MPSMatrix alloc] initWithBuffer:mpsBuffC descriptor:matDescriptorC];
    auto matC = [[MPSMatrix alloc] initWithBuffer:buffC descriptor:matDescriptorC];
    
    auto kernel = [[MPSMatrixMultiplication alloc] initWithDevice:gpu::device transposeLeft:NO transposeRight:NO resultRows:M resultColumns:P interiorColumns:N alpha:1.0 beta:1.0];
    
    constexpr auto ITERS = 8;
    double mpsTotalTime = 0.0;
    for (int i = 0; i < ITERS; i++) {
//    while (true) {
        memset(mpsBuffC.contents, 0, mpsBuffC.length);
        auto cmd = [gpu::queue commandBuffer];
        [kernel encodeToCommandBuffer:cmd leftMatrix:matA rightMatrix:matB resultMatrix:mpsMatC];
        [cmd commit];
        [cmd waitUntilCompleted];
        auto cmdTime = [cmd GPUEndTime] - [cmd GPUStartTime];
        mpsTotalTime += cmdTime;
        
        for (auto i = 0; i < vdspC->size(); i++)
            assert(vdspC->at(i) == ((float*)mpsBuffC.contents)[i]);
    }
    printf("MPS: AVG %.3f TFLOP/s\n", (double)flops / (mpsTotalTime / (double)ITERS) * 1e-12);

    // while (true) {
    double gemmTotalTime = 0.0;
    for (int i = 0; i < ITERS; i++) {
        memset(buffC.contents, 0, buffC.length);
        auto cmd = [gpu::queue commandBuffer];
        gpugemm::encode(cmd, matA, matB, matC);
        [cmd commit];
        [cmd waitUntilCompleted];
        auto cmdTime = [cmd GPUEndTime] - [cmd GPUStartTime];
        gemmTotalTime += cmdTime;
        
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
    printf("SGEMM: AVG %.3f TFLOP/s\n", (double)flops / (gemmTotalTime / (double)ITERS) * 1e-12);
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

int main_dot()
{
    constexpr auto N = 1 << 22;
    
    auto* x = new std::array<float, N>{};
    auto* y = new std::array<float, N>{};
    float vdspResult = 0.0;
    
    float naiveResult = 0.0;
    float gpuResult = 0.0;
    float mpsResult = 0.0;

    std::normal_distribution<float> dstr(0.0, 5.0);
    std::mt19937 gen{};
    
    for (int i = 0; i < N; i++) {
        x->at(i) = dstr(gen);
        y->at(i) = dstr(gen);
    }

    auto flops = N * 2ul;
    double gbs = (double)N * sizeof(float) / 1000 / 1000 / 1000;
    
    printf("TOTAL MFLOP %.3f\n", flops / 1e6);
    
    constexpr auto ITERS = 100;
    double vDSPTT = 0.0;
    for (auto i = 0; i < ITERS; i++) {
        vdspResult = 0;
        clock_t st = clock();
        vDSP_dotpr(x->data(), 1, y->data(), 1, &vdspResult, N);
        clock_t et = clock();
        vDSPTT += (double)(et - st) / CLOCKS_PER_SEC;
    }
    printf("vDSP AVG GFLOPS %.3f\n", (double)flops * 1e-9 / (vDSPTT / ITERS));

    double naiveTT = 0.0;
    for (auto i = 0; i < ITERS; i++) {
        naiveResult = 0;
        clock_t st = clock();
        for (uint64_t i = 0; i < N; i++) naiveResult += x->at(i) * y->at(i);
        clock_t et = clock();
        naiveTT += (double)(et - st) / CLOCKS_PER_SEC;
        
    }
    printf("naive AVG GFLOPS %.3f\n", (double)flops * 1e-9 / (naiveTT / ITERS));

    double gpuTT = 0.0;
    auto buffX = [gpu::device newBufferWithBytes:x->data() length:x->size() * sizeof(x->front()) options:MTLResourceStorageModeShared];
    auto buffY = [gpu::device newBufferWithBytes:y->data() length:y->size() * sizeof(y->front()) options:MTLResourceStorageModeShared];
    for (auto i = 0; i < ITERS; i++) {
//    while (true) {
        auto buffResult = [gpu::device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
        auto cmd = [gpu::queue commandBuffer];
        
        gpudot::encode2(cmd, buffX, buffY, buffResult);
        
        [cmd commit];
        [cmd waitUntilCompleted];
        
        gpuTT += ([cmd GPUEndTime] - [cmd GPUStartTime]);
        
        gpuResult = *(float*)buffResult.contents;
    }
    printf("gpu AVG GFLOPS %.3f\n", (double)flops * 1e-9 / (gpuTT / ITERS));
    printf("gpu AVG BANDWIDTH %.3f GB/s\n", gbs / (gpuTT / ITERS));
    printf("gpu AVG time %f ms\n", (gpuTT / ITERS) * 1000);
    
    if (false) {
        auto kernel = [[MPSMatrixMultiplication alloc] initWithDevice:gpu::device resultRows:1 resultColumns:1 interiorColumns:x->size()];

        auto buffX = [gpu::device newBufferWithBytes:x->data() length:x->size() * sizeof(x->front()) options:MTLResourceStorageModeShared];
        auto buffY = [gpu::device newBufferWithBytes:y->data() length:y->size() * sizeof(y->front()) options:MTLResourceStorageModeShared];
        auto buffResult = [gpu::device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
        auto cmd = [gpu::queue commandBuffer];
        
        auto xDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:1 columns:x->size() rowBytes:x->size() * sizeof(x->front()) dataType:MPSDataTypeFloat32];
        auto yDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:y->size() columns:1 rowBytes:sizeof(x->front()) dataType:MPSDataTypeFloat32];
        auto rDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:1 columns:1 rowBytes:sizeof(x->front()) dataType:MPSDataTypeFloat32];
        auto xMat = [[MPSMatrix alloc] initWithBuffer:buffX descriptor:xDesc];
        auto yMat = [[MPSMatrix alloc] initWithBuffer:buffY descriptor:yDesc];
        auto rMat = [[MPSMatrix alloc] initWithBuffer:buffResult descriptor:rDesc];
        [kernel encodeToCommandBuffer:cmd leftMatrix:xMat rightMatrix:yMat resultMatrix:rMat];
        
        [cmd commit];
        [cmd waitUntilCompleted];
        
        printf("mps GFLOPS %.3f\n", (double)flops * 1e-9 / ([cmd GPUEndTime] - [cmd GPUStartTime]));
        
        mpsResult = *(float*)buffResult.contents;
    }

    
    printf("\n%f %f %f %f\n", naiveResult, vdspResult, gpuResult, mpsResult);
    assert(fabsf(naiveResult - vdspResult) < 2);
    assert(fabsf(gpuResult - vdspResult) < 2);
//    assert(fabsf(mpsResult - vdspResult) < 2);

    return 0;
}

int main()
{
    printf("%lu\n", [gpu::device maxThreadgroupMemoryLength]);
    return main_dot();
}
