cmake_minimum_required(VERSION 3.1)

project(ch5)

set(CMAKE_BUILD_TYPE "Release")

add_subdirectory(imageBasics)
add_subdirectory(stereo)
add_subdirectory(rgbd)
add_subdirectory(calibration)
