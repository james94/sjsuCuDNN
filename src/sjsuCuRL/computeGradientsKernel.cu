#include "sjsuCuRL/computeGradientsKernel.cuh"

__global__ void computeGradientsKernel(const float* states, const int* actions, const float* returns, int length,
                                       PolicyNetwork* policy_net, float* gradients) {

}
