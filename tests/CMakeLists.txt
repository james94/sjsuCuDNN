cmake_minimum_required(VERSION 3.5)
project(sjsuCuDNNTests LANGUAGES CXX CUDA)

find_package(GTest REQUIRED)

# Add sjsuCuDNNTests executable
add_executable(${PROJECT_NAME}
    test.cpp
    testRelu.cu
    testReluKernel.cu
)

# Link against GTest
# message("CONAN_TARGETS = ${CONAN_TARGETS}")
target_link_libraries(${PROJECT_NAME}
    PUBLIC
    ${GTEST_LIBRARIES}
    ${CUDA_LIBRARIES}
    sjsuCuDNN
    # ${CONAN_TARGETS}
    # CONAN_PKG::gtest
)

target_include_directories(${PROJECT_NAME}
    PUBLIC
    ${GTEST_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    # ${CONAN_INCLUDE_DIRS}
)

enable_testing()

# Add sjsuCuDNNTests executable as a test
add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})