#include <gtest/gtest.h>
#include "sjsuCuDNN/linear.h"

TEST(LinearTest, OutputValidation) {
    const int size = 10;

    // Initialize input data on the host
    int batch_size = 0;
    int input_size = 0;
    int output_size = 0;

    float* host_input = nullptr;
    float* host_weight = nullptr;
    float* host_bias = nullptr;
    float* host_output = new float[batch_size * output_size];

    linear(host_input, host_weight, host_bias, host_output, batch_size, input_size, output_size);

    // Free allocated memory on host
    delete[] host_input;
    delete[] host_weight;
    delete[] host_bias;
    delete[] host_output;
}