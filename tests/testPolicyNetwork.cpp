#include <gtest/gtest.h>
#include "sjsuCuDNN/policyNetwork.h"
#include <opencv2/opencv.hpp>

TEST(SoftmaxTest, OutputValidation) {
    // Create a random 2D image
    cv::Mat image(28, 28, CV_32FC1);
    cv::randu(image, 0, 1);

    // Convert the image to a float array
    float* image_data = image.ptr<float>(0);

    int input_size = 784;
    int hidden_size = 256;
    int output_size = 10;

    PolicyNetwork network(input_size, hidden_size, output_size);

    float* action_probs = new float[output_size];
    network.forward(image_data, action_probs);

    // Validate the action_probs
    // For example, check if the sum of action_probs is close to 1
    float sum_probs = 0.0f;
    for(int i = 0; i < output_size; ++i) {
        sum_probs += action_probs[i];
    }
    ASSERT_TRUE(sum_probs, 1.0f, 1e-6);

    delete[] action_probs;
}