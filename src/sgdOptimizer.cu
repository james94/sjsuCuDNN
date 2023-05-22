#include <sjsuCuDNN/sgdUpdateKernel.cuh>
#include <sjsuCuDNN/sgdOptimizer.cuh>

void SGDOptimizer::step() {
    sgdUpdate();
}

/*
    PolicyNetwork& policy_net_;
    float learning_rate_;
*/
void SGDOptimizer::sgdUpdate() {
    const std::vector<float>& gradients = policy_net_.getGradients(); // no method yet

    // Allocate device memory for parameters and gradients
    float* d_parameters;
    float* d_gradients;
    cudaMalloc(&d_parameters, gradients.size() * sizeof(float));
    cudaMalloc(&d_gradients, gradients.size() * sizeof(float));

    // Transfer data from host to device
    // no getParameters() method yet
    cudaMemcpy(d_paramters, policy_net_.getParameters(), gradients.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gradients, gradients.data(), gradients.size() * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the SGD update kernel
    int blockSize = 256;
    int numBlocks = (gradients.size() + blockSize - 1) / blockSize;
    sgdUpdateKernel<<<numBlocks, blockSize>>>(d_parameters, d_gradients, learning_rate_, gradients.size());

    // Transfer updated parameters from device back to host
    cudaMemcpy(policy_net_.getParameters(), d_parameters, gradients.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_parameters);
    cudaFree(d_gradients);
}
