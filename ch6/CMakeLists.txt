cmake_minimum_required(VERSION 3.1)
project(ch6)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED on)

#list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# OpenCV
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

# g2o
#find_package(G2O REQUIRED)
#include_directories(${G2O_INCLUDE_DIRS})

# Eigen
include_directories("c:/Users/qkrwh/3rdparty/install/eigen3/include/eigen3")

add_executable(gaussNewton gaussNewton.cpp)
target_link_libraries(gaussNewton ${OpenCV_LIBS})

add_executable(ceresCurveFitting ceresCurveFitting.cpp)
target_link_libraries(ceresCurveFitting ${OpenCV_LIBS} ${CERES_LIBRARIES})

#add_executable(g2oCurveFitting g2oCurveFitting.cpp)
#target_link_libraries(g2oCurveFitting ${OpenCV_LIBS} ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY})
