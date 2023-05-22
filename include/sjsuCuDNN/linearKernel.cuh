#include <cuda.h>
#include <cuda_runtime.h>

#ifndef SJSU_CUDNN_LINEAR_KERNEL_CUH
#define SJSU_CUDNN_LINEAR_KERNEL_CUH

__global__ void linearKernel(const float* input, const float* weight, const float* bias,
                             float* output, int input_size int output_size);

#endif // SJSU_CUDNN_LINEAR_KERNEL_CUH
