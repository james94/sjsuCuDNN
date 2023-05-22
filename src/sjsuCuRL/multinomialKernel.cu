#include "sjsuCuRL/multinomialKernel.cuh"

__global__ void multinomialKernel(const float* action_probs, int num_actions, int num_samples, int* actions) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < num_samples) {
        float rand_val = curand_uniform(&curand_state[index]);
        float cum_prob = 0.0;
        for(int action = 0; action < num_actions; ++action) {
            cum_prob += action_probs[action];
            if(rand_val < cum_prob) {
                actions[index] = action;
                break;
            }
        }
    }
}