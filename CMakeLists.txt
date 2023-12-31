cmake_minimum_required(VERSION 3.10)

project(PPN-CANCER-IA)

set(CMAKE_CXX_STANDARD 17)

# Enable debugging information
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

set(SOURCE_FILES
    src/main.cpp
    src/image.cpp
    src/tools.cpp
    src/conv_op.cpp
    src/network.cpp
    src/topo.cpp
	src/layers/layer.cpp
    src/layers/convolution.cpp
    src/layers/pooling.cpp
    src/layers/dense.cpp
    src/layers/output.cpp
	src/layers/activations/activation.cpp
    src/layers/activations/relu.cpp
    src/layers/activations/softmax.cpp
)

set(HEADER_FILES
    include/image.h
    include/tools.h
    include/conv_op.h
    include/network.h
    include/topo.h
	include/layers/layer.h
    include/layers/convolution.h
    include/layers/pooling.h
    include/layers/output.h
    include/layers/dense.h
	include/layers/activations/activation.h
    include/layers/activations/relu.h
    include/layers/activations/softmax.h
)

add_executable(main.exe ${SOURCE_FILES})

target_include_directories(main.exe PRIVATE
    "${PROJECT_SOURCE_DIR}/include"
)

target_include_directories(main.exe PRIVATE
    "${PROJECT_SOURCE_DIR}/include/layers"
)

target_include_directories(main.exe PRIVATE
    "${PROJECT_SOURCE_DIR}/include/layers/activations"
)

find_package(PNG REQUIRED)
target_link_libraries(main.exe PRIVATE PNG::PNG)
