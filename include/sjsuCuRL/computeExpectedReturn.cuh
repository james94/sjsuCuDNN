#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

#ifndef SJSU_CURL_COMPUTE_EXPECTED_RETURN_KERNEL_CUH
#define SJSU_CURL_COMPUTE_EXPECTED_RETURN_KERNEL_CUH

__device__ void computeExpectedReturn(const float* rewards, int length);

#endif // SJSU_CURL_COMPUTE_EXPECTED_RETURN_KERNEL_CUH
