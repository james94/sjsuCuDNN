#include "sjsuCuRL/computeExpectedReturn.cuh"
#include "sjsuCuRL/computeLossKernel.cuh"

__global__ void computeLossKernel(const float* states, const int* actions, const float* rewards, 
    float* action_probs, float* expected_return, float* loss, int num_samples, PolicyNetwork* policy_net,
    int input_size, int output_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid < num_samples) {
        // Get current sample's data
        const float* state = states + tid * input_size;
        int action = actions[tid];
        float reward = rewards[tid];

        // Forward pass to compute the action probabilities
        network.forward(state, action_probs + tid * output_size);

        // Compute the negative log-likelihodd of the chosen action
        float neg_log_prob = -logf(action_probs[tid * output_size + action]);

        // Compute the expected return
        expected_return[tid] = computeExpectedReturn(rewards + tid, num_samples - tid);

        // Compute the loss
        loss[tid] = neg_log_prob * expected_return[tid]; 
    }
}
