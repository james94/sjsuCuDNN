#ifndef SJSU_CUDNN_POLICY_NETWORK_H
#define SJSU_CUDNN_POLICY_NETWORK_H

class PolicyNetwork {
public:
    PolicyNetwork(int input_size, int hidden_size, int output_size);

    virtual ~PolicyNetwork();

    void forward(const float* input, float* action_probs);
private:
    void linearLayer(const float* input, const float* weight, const float* bias,
                     float* output, int batch_size, int input_size int output_size);

    void reluActivation(const float* input, float* output, int size);

    void softmaxActivation(const float* input, float* output, int size);

    float* weight1_;
    float* bias1_;
    float* weight2_;
    float* bias2_;

    int input_size_;
    int hidden_size_;
    int output_size_;
};

#endif // SJSU_CUDNN_POLICY_NETWORK_H
