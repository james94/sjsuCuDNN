#include "sjsuCuDNN/softmaxKernel.cuh"

__global__ void softmaxKernel(const float* input, float* output, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index < size) {
        float max_val = input[index];
        for(int i = 0; i < size; i++) {
            max_val = max(max_val, input[i]);
        }

        float sum_exp = 0.0f;
        for(int i = 0; i < size; i++) {
            sum_exp += expf(input[i] - max_val);
        }

        output[index] = expf(input[index] - max_val) / sum_exp;
    }
}
