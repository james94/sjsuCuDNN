#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

#ifndef SJSU_CURL_COMPUTE_GRADIENTS_KERNEL_CUH
#define SJSU_CURL_COMPUTE_GRADIENTS_KERNEL_CUH

__global__ void computeGradientsKernel(const float* states, const int* actions, const float* returns, int length,
                                       PolicyNetwork* policy_net, float* gradients);

#endif // SJSU_CURL_COMPUTE_GRADIENTS_KERNEL_CUH
