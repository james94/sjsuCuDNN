#include <iostream>
#include <vector>
#include "policyNetwork.cuh"

class SGDOptimizer {
public:
    explicit SGDOptimizer(PolicyNetwork& policy_net, float learning_rate)
        : policy_net_(policy_net), learning_rate_(learning_rate) {}

    void step();

private:
    void sgdUpdate();

    PolicyNetwork& policy_net_;
    float learning_rate_;
}