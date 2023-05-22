cmake_minimum_required(VERSION 3.5)
project(sjsuCuDNN LANGUAGES CXX CUDA)

# include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
# conan_basic_setup()

enable_testing()

function(find_cuda)
    if(UNIX)
        # Search for CUDA on Ubuntu
        find_package(CUDA REQUIRED)
        set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})
    elseif(WIN32)
        # Search for CUDA on Windows
        if(NOT DEFINED ENV{CUDA_PATH})
            message(FATAL_ERROR "CUDA_PATH environment variable not set")
        endif()
        set(CUDA_PATH $ENV{CUDA_PATH})
    else()
        message(FATAL_ERROR "Unsupported platform")
    endif()

    if(NOT CUDA_FOUND)
        message(FATAL_ERROR "CUDA not found")
    endif()

    set(CMAKE_PREFIX_PATH ${CUDA_PATH} ${CMAKE_PREFIX_PATH})
    message(STATUS "CUDA found at: ${CUDA_PATH}")
endfunction()

find_cuda()

# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11" )

# link_directories(/usr/local/cuda/lib64)

# set(CMAKE_CXX_COMPILER "/usr/bin/g++-8")

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
# set(CMAKE_CUDA_STANDARD 11)

set(SRCS
    src/relu.cu
    src/reluKernel.cu
)

set(HDRS
    include/sjsuCuDNN/relu.cuh
    include/sjsuCuDNN/reluKernel.cuh
)

# Add CUDA include directories
# include_directories(${CUDA_INCLUDE_DIRS})

cuda_add_library(sjsuCuDNN ${SRCS})

target_include_directories(sjsuCuDNN PUBLIC 
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(sjsuCuDNN 
    ${CUDA_LIBRARIES}
    cudart
)

set_target_properties(sjsuCuDNN PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(sjsuCuDNN PROPERTIES PUBLIC_HEADER "${HDRS}")

add_subdirectory(tests)