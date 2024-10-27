cmake_minimum_required( VERSION 2.8 )
project( geometry )

set(CMAKE_BUILD_TYPE "Release")

include_directories( "../../3rdparty/eigen3")

add_executable(eigenGeometry eigenGeometry.cpp)