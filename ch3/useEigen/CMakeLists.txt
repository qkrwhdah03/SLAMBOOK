cmake_minimum_required(VERSION 2.8)
project(useEigen)

set(CMAKE_BUILD_TYPE "Release")

include_directories("c:/Users/qkrwh/3rdparty/install/eigen3/include/eigen3")

add_executable(eigenMatrix eigenMatrix.cpp)
