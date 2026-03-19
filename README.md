# gemm
GEMM is General Matrix Multiply. This project is aimed to implement gemm algorithm on cpu and gpu and iteratively enhance them to achieve maximum reference speed.

For CPU maximum reference speed is numpy (although there is a comparison with vDSP, but it's unfair since vDSP uses private API matrix cores)

For GPU maximum reference speed is MPS

### CPU benchmarks
```console
$ python3 gemm.py                                                                                                                                                                               
GFLOP 17.180
numpy AVG GFLOPS 113.727

$ ./buildrun.sh                                                                                                                                                                                 
GFLOP 17.180
vdsp GFLOPS 868.065
gemm AVG GFLOPS 111.166
```

### GPU benchmarks
```console
$ cd gpugemm && ./buildrun.sh
TFLOP 0.137
MPS: AVG 7.473 TFLOP/s
SGEMM: AVG 7.152 TFLOP/s
```

##### gemv GPU benchmarks (bonus)
gpugemm also have a gemv implementation (general matrix vector multiply)
```console
$ cd gpugemm && ./buildrun.sh
GFLOP 0.268
MPS:   AVG 179.746 GFLOP/s
SGEMV: AVG 188.408 GFLOP/s
```
> to run this benchmark please change main to call main_vec

# Conclussion
This is a very fun experience to implement such foundational algorithm such as gemm with perfomance near to SOT implementations such as numpy and MPS

gemv was especially fun because I think its really better then MPS but need to double check

unrolling on gpu is insane, sgemm32x32_unrolled is basically manually unrolled version of sgemm32x32 which is x5 faster with no more modifications

All code in this project is written manually, by hand, purely for recreational and educations purposes
