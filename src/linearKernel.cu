#include "sjsuCuDNN/linearKernel.cuh"

__global__ void linearKernel(const float* input, const float* weight, const float* bias,
                             float* output, int input_size int output_size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index < output_size) {
        float sum = 0.0f;
        for(int i = 0; i < input_size; i++) {
            sum += input[i] * weight[index * input_size + i];
        }
        output[index] = sum + bias[index];
    }
}