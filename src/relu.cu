#include <sjsuCuDNN/reluKernel.cuh>
#include <sjsuCuDNN/relu.h>
#include <iostream>

void relu(const float* host_input, float* host_output, int size) {
    const int threads_per_block = 256;
    const int num_blocks = (size + threads_per_block - 1) / threads_per_block;

    // Allocate memory on the device for input and output
    float* device_input;
    float* device_output;
    cudaMalloc(&device_input, size * sizeof(float));
    cudaMalloc(&device_output, size * sizeof(float));

    // Transfer input from host to device using cudaMemcpy
    cudaMemcpy(device_input, host_input, size * sizeof(float), cudaMemcpyHostToDevice);
    // Ensure that all data transfers to device are done before proceeding
    cudaDeviceSynchronize();
    // Launch the CUDA kernel
    reluKernel<<<num_blocks, threads_per_block>>>(device_input, device_output, size);
    // Transfer the output data from device to host using cudaMemcpy
    cudaMemcpy(host_output, device_output, size * sizeof(float), cudaMemcpyDeviceToHost);
    // Ensure data transfer to host is done before proceeding
    cudaDeviceSynchronize();

    // Free allocated memory on the device
    cudaFree(device_input);
    cudaFree(device_output);
}
