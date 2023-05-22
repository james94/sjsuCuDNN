#include "sjsuCuRL/computeLossKernel.cuh"
#include "sjsuCuRL/computeLoss.cuh"

void computeLoss(const float* states, const int* actions, const float* rewards, 
    float* action_probs, float* expected_return, float* loss, int num_samples, 
    PolicyNetwork* policy_net, int input_size, int output_size) {
    int block_size = 256;
    int grid_size = (num_samples + block_size - 1) / block_size;

    computeLossKernel<<<grid_size, block_size>>>(states, actions, rewards, action_probs,
        expected_return, loss, num_samples, policy_net, input_size, output_size);

    cudaDeviceSynchronize();
}
