cmake_minimum_required(VERSION 3.1)

project(calibration)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED on)

add_executable(calibration calibration.cpp)

# Eigen
include_directories("c:/Users/qkrwh/3rdparty/install/eigen3/include/eigen3")

# OpenCv
set(OpenCV_DIR "c:/Users/qkrwh/3rdparty/install/opencv")
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV_DIR : ${OpenCV_DIR}")
    message(STATUS "OpenCV_INCLUDE_DIRS : ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "OpenCV_LIBS : ${OpenCV_LIBS}")
endif()
include_directories(${OpenCV_INCLUDE_DIRS})

# Ceres
set(Ceres_DIR "c:/Users/qkrwh/3rdparty/install/ceres-solver/lib/cmake/Ceres/")
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

target_link_libraries(calibration ${OpenCV_LIBS} ${CERES_LIBRARIES})