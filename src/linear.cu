#include <sjsuCuDNN/linearKernel.cuh>
#include <sjsuCuDNN/linear.h>
#include <iostream>

void linear(const float* input, const float* weight, const float* bias,
            float* output, int batch_size, int input_size int output_size) {
    const int threads_per_block = 256;
    const int num_blocks = (output_size + threads_per_block - 1) / threads_per_block;

    // Allocate memory on the device for input, weight, bias and output
    float* device_input;
    float* device_weight;
    float* device_bias;
    float* device_output;
    cudaMalloc(&device_input, batch_size * input_size * sizeof(float));
    cudaMalloc(&device_weight, input_size * output_size * sizeof(float));
    cudaMalloc(&device_bias, output_size * sizeof(float));
    cudaMalloc(&device_output, batch_size * output_size * sizeof(float));

    // Transfer input, weight and bias from host to device using cudaMemcpy
    cudaMemcpy(device_input, input, batch_size * input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_weight, weight, input_size * output_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_bias, bias, output_size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the CUDA kernel
    linearKernel<<<num_blocks, threads_per_block>>>(device_input, device_weight, device_bias,
                                                    device_output, input_size, output_size);

    // Transfer the output from device to host 
    cudaMemcpy(output, device_output, batch_size * output_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free allocated memory on the device
    cudaFree(device_input);
    cudaFree(device_weight);
    cudaFree(device_bias);
    cudaFree(device_output);
}
