#include <sjsuCuDNN/linearKernel.cuh>
#include <sjsuCuDNN/linear.h>
#include <sjsuCuDNN/reluKernel.cuh>
#include <sjsuCuDNN/relu.h>
#include <sjsuCuDNN/softmaxKernel.cuh>
#include <sjsuCuDNN/softmax.h>
#include <iostream>

PolicyNetwork::PolicyNetwork(int input_size, int hidden_size, int output_size) {
    this->input_size_ = input_size;
    this->hidden_size_ = hidden_size;
    this->output_size_ = output_size;

    // Initialize weights and biases
    cudaMalloc(&weight1, hidden_size * input_size * sizeof(float));
    cudaMalloc(&bias1, hidden_size * sizeof(float));
    cudaMalloc(&weight2, output_size * hidden_size * sizeof(float));
    cudaMalloc(&bias2, output_size * sizeof(float));

    // Initialize weights and biases with random values

    // Transfer weights and biases from host to device
}

virtual ~PolicyNetwork::PolicyNetwork() {
    cudaFree(weight1_);
    cudaFree(bias1_);
    cudaFree(weight2_);
    cudaFree(bias2_);
}

void PolicyNetwork::forward(const float* input, float* action_probs) {
    float* hidden = new float[hidden_size_];
    float* logits = new float[output_size_];

    // Perform Linear Transformation1; 1 for batch size
    linearLayer(input, weight1_, bias1_, hidden, 1, input_size_, hidden_size_);
    reluActivation(hidden, hidden, hidden_size_);

    // Perform Linear Transformation2
    linearLayer(hidden, weight2_, bias2_, logits, 1, hidden_size_, output_size_);
    softmaxActivation(logits, hidden, output_size_)

    delete[] hidden;
    delete[] logits;
}


void PolicyNetwork::linearLayer(const float* input, const float* weight, const float* bias,
                     float* output, int batch_size, int input_size int output_size) {
    linear(input, weight, bias, output, batch_size, input_size, output_size);
}

void PolicyNetwork::reluActivation(const float* input, float* output, int size) {
    relu(input, output, size);
}

void PolicyNetwork::softmaxActivation(const float* input, float* output, int size) {
    softmax(input, output, size);
}
