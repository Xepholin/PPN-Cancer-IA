cmake_minimum_required(VERSION 3.10)

project(PPN-CANCER-IA)

set(CMAKE_CXX_STANDARD 17)

set(SOURCE_FILES
    src/main.cpp
    src/matrice.cpp
)

set(HEADER_FILES
    include/matrice.h
)

add_executable(main.exe ${SOURCE_FILES})

target_include_directories(main.exe PRIVATE
    "${PROJECT_SOURCE_DIR}/include"
)
