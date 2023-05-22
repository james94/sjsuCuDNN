#ifndef SJSU_CUDNN_LINEAR_H
#define SJSU_CUDNN_LINEAR_H

void linear(const float* input, const float* weight, const float* bias,
            float* output, int batch_size, int input_size int output_size);

#endif // SJSU_CUDNN_LINEAR_H