#include "sjsuCuDNN/sgdUpdateKernel.cuh"

__global__ void sgdUpdateKernel(float* parameters, const float* gradients, float learning_rate, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index < size) {
        parameters[index] -= learning_rate * gradients[index];
    }
}
