#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

#ifndef SJSU_CURL_COMPUTE_LOSS_CUH
#define SJSU_CURL_COMPUTE_LOSS_CUH

void computeLoss(const float* states, const int* actions, const float* rewards, 
    float* action_probs, float* expected_return, float* loss, int num_samples, 
    PolicyNetwork* policy_net, int input_size, int output_size);

#endif // SJSU_CURL_COMPUTE_LOSS_CUH
