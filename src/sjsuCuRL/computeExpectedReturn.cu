#include "sjsuCuRL/computeExpectedReturnKernel.cuh"

__device__ void computeExpectedReturn(const float* rewards, int length) {
    float cumulative_return = 0.0f;
    for(int i = length - 1; i >= 0; --i) {
        cumulative_return = rewards[i] + cumulative_return;
    }
    return cumulative_return;
}
