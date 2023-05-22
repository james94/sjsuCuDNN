#include <cuda.h>
#include <cuda_runtime.h>

#ifndef SJSU_CUDNN_SGD_UPDATE_KERNEL_CUH
#define SJSU_CUDNN_SGD_UPDATE_KERNEL_CUH

__global__ void sgdUpdateKernel(float* parameters, const float* gradients, float learning_rate, int size);

#endif // SJSU_CUDNN_SGD_UPDATE_KERNEL_CUH