#include <sjsuCuDNN/softmaxKernel.cuh>
#include <sjsuCuDNN/softmax.h>
#include <iostream>

void softmax(const float* host_input, float* host_output, int size) {
    const int threads_per_block = 256;
    const int num_blocks = (size + threads_per_block - 1) / threads_per_block;

    // Allocate memory on the device for input and output
    float* device_input;
    float* device_output;
    cudaMalloc(&device_input, size * sizeof(float));
    cudaMalloc(&device_output, size * sizeof(float));

    // Transfer input from host to device
    cudaMemcpy(device_input, host_input, size * sizeof(float), cudaMemcpyHostToDevice);

    softmaxKernel<<<num_blocks, threads_per_block>>>(device_input, device_output, size);

    // Transfer the output from device to host
    cudaMemcpy(host_output, device_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);
}
