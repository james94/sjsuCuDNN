#include <gtest/gtest.h>
#include "sjsuCuRL/policyNetwork.cuh"
#include "sjsuCuRL/computeLossKernel.cuh"
#include "sjsuCuRL/computeLoss.cuh"
#include <opencv2/opencv.hpp>

// Generate a random 2D image using OpenCV
cv::Mat generateRandomImage(int width, int height) {
    cv::Mat image(height, width, CV_8UC1);
    cv::randu(image, cv::Scalar(0), cv::Scalar(255));
    return image;
}

TEST(ComputeLossTest, LossAndRewardsValidation) {
    // Generate example data
    const int num_samples = 10;
    const int input_size = 784;
    const int hidden_size = 128;
    const int output_size = 10;

    cv::Mat image = generateRandomImage(28, 28);

    // Convert image to float array
    cv::Mat imageFloat;
    image.convertTo(imageFloat, CV_32F);

    // Generate rnadom states, actions, and rewards
    float* states = new float[num_samples * input_size];
    int* actions = new int[num_samples];
    float* rewards = new float[num_samples];

    for(int i = 0; i < num_samples * input_size; ++i) {
        states[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    for(int i = 0; i < num_samples; ++i) {
        actions[i] = rand() % output_size;
        rewards[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    PolicyNetwork policyNetwork(input_size, hidden_size, output_size);
    float* action_probs = new float[output_size];

    // Compute loss also using computeExpectedReturn
    float* loss = new float[num_samples];
    float* expected_return = new float[num_samples];
    ComputeLoss(states, actions, rewards, action_probs, expected_return, loss, num_samples, 
        policyNetwork, input_size, output_size);

    // Validate loss and rewards
    for(int i = 0; i < num_samples; ++i) {
        // Validate loss
        EXPECT_GE(loss[i], 0.0f);

        // Validate rewards
        EXPECT_GE(rewards[i], 0.0f);
    }

    delete[] states;
    delete[] actions;
    delete[] rewards;
    delete[] loss;
}
