#include <gtest/gtest.h>
#include "sjsuCuDNN/reluKernel.cuh"

TEST(ReluKernelTest, OutputValidation) {
    const int size = 10;
    const int block_size = 256;

    float* input;
    float* output;
    
    // Allocate memory on the device
    cudaMalloc(&input, size * sizeof(float));
    cudaMalloc(&output, size * sizeof(float));

    // Initialize input data on the host
    float host_input[size] = {-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0, 10.0};

    // Copy input data from host to device
    cudaMemcpy(input, host_input, size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    reluKernel<<<(size + block_size - 1) / block_size, block_size>>>(input, output, size);

    // Copy output data from device to host
    float host_output[size];
    cudaMemcpy(host_output, output, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Validate the output
    float expected_output[size] = {0.0, 2.0, 0.0, 4.0, 0.0, 6.0, 0.0, 8.0, 0.0, 10.0};
    for(int i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(host_output[i], expected_output[i]);
    }

    // Free allocated memory on the device
    cudaFree(input);
    cudaFree(output);
}
