#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#ifndef SJSU_CUDNN_SOFTMAX_KERNEL_CUH
#define SJSU_CUDNN_SOFTMAX_KERNEL_CUH

__global__ void softmaxKernel(const float* input, float* output, int size);

#endif // SJSU_CUDNN_SOFTMAX_KERNEL_CUH
