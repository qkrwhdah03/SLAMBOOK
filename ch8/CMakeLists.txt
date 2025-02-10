cmake_minimum_required(VERSION 3.1)
project(ch8)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED on)

# Eigen
include_directories("c:/Users/qkrwh/3rdparty/install/eigen3/include/eigen3")

# OpenCv
set(OpenCV_DIR "c:/Users/qkrwh/3rdparty/install/opencv")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Sophus
set(Sophus_DIR "c:/Users/qkrwh/3rdparty/install/Sophus/share/sophus/cmake")
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIR})

# Threads 
find_package(Threads REQUIRED)

# GLEW 
set(GLEW_DIR "c:/Users/qkrwh/3rdparty/install/glew/lib/cmake/glew")
find_package(GLEW REQUIRED)
include_directories(${GLEW_INCLUDE_DIRS})

# Pangolin
set(Pangolin_DIR "c:/Users/qkrwh/3rdparty/install/Pangolin/lib/cmake/Pangolin")
find_package(Pangolin REQUIRED)


add_executable(optical_flow optical_flow.cpp)
target_link_libraries(optical_flow ${OpenCV_LIBS})

add_executable(direct_method direct_method.cpp)
target_link_libraries(direct_method ${OpenCV_LIBS} ${Pangolin_LIBRARIES} Sophus::Sophus)
