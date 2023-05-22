#include <gtest/gtest.h>
#include "sjsuCuRL/multinomialKernel.cuh"
#include "sjsuCuRL/policyNetwork.cuh"
#include <opencv2/opencv.hpp>

TEST(MultinomialSamplingTest, SampledActionsValidation) {
    // Define the hyperparameters
    int input_size = 4;
    int hidden_size = 64;
    int output_size = 6;
    int num_samples = 100;

    PolicyNetwork policy_net(input_size, hidden_size, output_size);

    // Generate example input data
    cv::Mat image = cv::Mat::zeros(input_size, 1, CV_32FC1);
    float* state = image.ptr<float>();

    // Allocate device memory for action porobabilities and actions
    float* d_state;
    float* d_action_probs;

    cudaMalloc(&d_state, input_size * sizeof(float));
    cudaMalloc(&d_action_probs, output_size * sizeof(float));

    // Transfer data from host to device
    cudaMemcpy(d_state, state, input_size * sizeof(float), cudaMemcpyHostToDevice);

    policy_net.forward(d_state, d_action_probs);

    // Allocate device memory for sampled actions
    int* d_actions;
    cudaMalloc(&d_actions, num_samples * sizeof(int));


    // Launch the multinomial sampling
    int blockSize = 256;
    int numBlocks = (num_samples + blockSize - 1) / blockSize;
    multinomialKernel<<<numBlocks, blockSize>>>(d_action_probs, output_size, num_samples, d_actions);

    // Transfer data from device to host
    int* sampled_actions = new int[num_samples];
    cudaMemcpy(sampled_actions, d_actions, num_samples * sizeof(int), cudaMemcpyDeviceToHost);

    // Validate the sampled actions
    for(int i = 0; i < num_samples; ++i) {
        ASSERT_GE(sampled_actions[i], 0);
        ASSERT_LT(sampled_actions[i], output_size);
    }

    // Free device memory
    cudaFree(d_state);
    cudaFree(d_action_probs);
    cudaFree(d_actions); 
    delete[] sampled_actions;
}