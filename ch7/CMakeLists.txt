cmake_minimum_required(VERSION 3.1)
project(vo1)

set(CMAKE_BUILD_TYPE "Release")
#add_definitions("-DENABLE_SSE")
#set(CMAKE_CXX_FLAGS "-std=c++11 -O2 ${SSE_FLAGS} -msse4")
#list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Eigen
include_directories("c:/Users/qkrwh/3rdparty/install/eigen3/include/eigen3")

# OpenCv
set(OpenCV_DIR "c:/Users/qkrwh/3rdparty/install/opencv")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Ceres
set(Ceres_DIR "c:/Users/qkrwh/3rdparty/install/ceres-solver/lib/cmake/Ceres/")
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

# Sophus
set(Sophus_DIR "c:/Users/qkrwh/3rdparty/install/Sophus/share/sophus/cmake")
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

#target_link_libraries(calibration ${OpenCV_LIBS} ${CERES_LIBRARIES})
#target_link_libraries(useSophus PUBLIC Sophus::Sophus)

add_executable(orb_cv orb_cv.cpp)
target_link_libraries(orb_cv ${OpenCV_LIBS})

#add_executable(orb_self orb_self.cpp)
#target_link_libraries(orb_self ${OpenCV_LIBS})


add_executable(pose_estimation_2d2d pose_estimation_2d2d.cpp)
target_link_libraries(pose_estimation_2d2d ${OpenCV_LIBS})

add_executable(triangulation triangulation.cpp)
target_link_libraries(triangulation ${OpenCV_LIBS})

#add_executable(pose_estimation_3d2d pose_estimation_3d2d.cpp)
#target_link_libraries(pose_estimation_3d2d g2o_core g2o_stuff ${OpenCV_LIBS})

#add_executable(pose_estimation_3d3d pose_estimation_3d3d.cpp)
#target_link_libraries(pose_estimation_3d3d g2o_core g2o_stuff ${OpenCV_LIBS})
