#include <cuda.h>
#include <cuda_runtime.h>

#ifndef SJSU_CUDNN_RELU_KERNEL_CUH
#define SJSU_CUDNN_RELU_KERNEL_CUH

__global__ void reluKernel(const float* input, float* output, int size);

#endif // SJSU_CUDNN_RELU_KERNEL_CUH