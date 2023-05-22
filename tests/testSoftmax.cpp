#include <gtest/gtest.h>
#include "sjsuCuDNN/softmax.h"

TEST(SoftmaxTest, OutputValidation) {
    const int size = 10;

    float* host_input = nullptr;
    float* host_output = new float[bsize];

    softmax(host_input, host_output, size);

    // Free allocated memory on host
    delete[] host_input;
    delete[] host_output;
}