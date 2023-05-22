#include <curand_kernel.h>

#ifndef SJSU_CURL_MULTINOMIAL_KERNEL_CUH
#define SJSU_CURL_MULTINOMIAL_KERNEL_CUH

__global__ void multinomialKernel(const float* action_probs, int num_actions, int num_samples, int* actions);

#endif // SJSU_CURL_MULTINOMIAL_KERNEL_CUH
