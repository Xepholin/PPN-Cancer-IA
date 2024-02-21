cmake_minimum_required(VERSION 3.10)

project(PPN-CANCER-IA)

set(CMAKE_CXX_STANDARD 17)

# Enable debugging information
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -mavx -march=native -O3")

# Main program source files
set(MAIN_SOURCE_FILES
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

# Header files
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

# Main program executable
add_executable(main.exe ${MAIN_SOURCE_FILES})

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



# # Include Google Test
# find_package(GTest REQUIRED)
# include_directories(${GTEST_INCLUDE_DIRS})

# # Test executable
# add_executable(test.exe src/conv_op.cpp src/layers/layer.cpp src/layers/pooling.cpp test/tests.cpp)


# target_include_directories(test.exe PRIVATE
#     "${PROJECT_SOURCE_DIR}/include"
# )

# target_include_directories(test.exe PRIVATE
#     "${PROJECT_SOURCE_DIR}/include/layers"
# )

# target_include_directories(test.exe PRIVATE
#     "${PROJECT_SOURCE_DIR}/include/layers/activations"
# )

# # Link PNG library to tests.exe (if needed)
# target_link_libraries(test.exe PRIVATE PNG::PNG)

# # GTest
# find_package(GTest REQUIRED)
# target_link_libraries(test.exe PUBLIC GTest::GTest GTest::Main)
# enable_testing()

# # Tests
# gtest_discover_tests(test.exe)