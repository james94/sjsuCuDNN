#include <gtest/gtest.h>
#include "sjsuCuDNN/relu.cuh"

TEST(ReluTest, OutputValidation) {
    const int size = 10;

    // Initialize input data on the host
    float host_input[size] = {-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0, 10.0};
    float host_output[size] = {0.0};

    relu(host_input, host_output, size);

    // std::cout << "Host Out:\n";
    // for(int i = 0; i < size; ++i) {
    //     std::cout << host_output[i] << " ";
    // }
    // std::cout << std::endl;

    // Validate the output
    float expected_output[size] = {0.0, 2.0, 0.0, 4.0, 0.0, 6.0, 0.0, 8.0, 0.0, 10.0};
    for(int i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(host_output[i], expected_output[i]);
    }
}
