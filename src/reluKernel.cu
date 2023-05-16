#include "sjsuCuDNN/reluKernel.cuh"

__global__ void reluKernel(const float* input, float* output, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index < size) {
        output[index] = max(0.0f, input[index]);
    }
}
