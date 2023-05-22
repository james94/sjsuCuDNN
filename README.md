# sjsuCuDNN

Open source CuDNN library created during CMPE 214 GPU CUDAPong project. 
However, this sjsuCuDNN library will be able to be integrated into other
AI projects and I plan to leverage it for my master thesis on Deep Learning
in AI Medical Imaging for Stroke Diagnosis. AI Stroke Diagnosis has features
for stroke lesion segmentation and stroke image captioning with potential
to extend toward stroke image report generation.

## Deep Learning

- Layers: linear, relu, softmax
    - CUDA Kernels: linearKernel, reluKernel, softmaxKernel

- Networks: PolicyNetwork with forward pass
    - CUDA Kernels: linearKernel, reluKernel, softmaxKernel

- GTESTs for CUDA Layers: testLinear, testRelu, testSoftmax

## Deep Reinforcement Learning

- Extending Policy Network to be Policy Gradient Network
    - RL CUDA Kernels: computeLossKernel, computeEpectedReturn, multinomialKernel
    - Working on adding a CUDA Stochastic Gradient Descent (SGD) Optimizer for updating
    the gradients of our Policy Gradient Network
    - Working on adding CUDA computeGradientsKernel

- GTEST for CUDA multinomial sampling actions: testMultinomialKernel
- GTEST for CUDA computeLoss for policy gradient network: testComputeLoss

## Build Instructions

Build sjsuCuDNN:

~~~bash
mkdir build
cd build
cmake ..
cmake --build .
~~~

Run relu and reluKernel GTEST

~~~bash
cd tests
./sjsuCuDNNTests
~~~

Example of the GTESTs output for relu and reluKernel:

~~~bash
(cmpe260_project) james@james-Zephyrus-S-GX531GW:~/github/james/sjsuCuDNN/build/tests$ ./sjsuCuDNNTests 
[==========] Running 2 tests from 2 test suites.
[----------] Global test environment set-up.
[----------] 1 test from ReluTest
[ RUN      ] ReluTest.OutputValidation
[       OK ] ReluTest.OutputValidation (1696 ms)
[----------] 1 test from ReluTest (1696 ms total)

[----------] 1 test from ReluKernelTest
[ RUN      ] ReluKernelTest.OutputValidation
[       OK ] ReluKernelTest.OutputValidation (0 ms)
[----------] 1 test from ReluKernelTest (0 ms total)

[----------] Global test environment tear-down
[==========] 2 tests from 2 test suites ran. (1696 ms total)
[  PASSED  ] 2 tests.
~~~