cmake_minimum_required(VERSION 3.1)
project(rgbd)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED on)

add_executable(joinMap joinMap.cpp)

# Sophus
set(Sophus_DIR "c:/Users/qkrwh/3rdparty/install/Sophus/share/sophus/cmake")
find_package(Sophus REQUIRED)

if(Sophus_FOUND)
    message(STATUS "Sophus_DIR : ${Sophus_DIR}")
endif()
target_link_libraries(joinMap Sophus::Sophus)

# Eigen
include_directories("c:/Users/qkrwh/3rdparty/install/eigen3/include/eigen3")

# Threads 
find_package(Threads REQUIRED)

# GLEW 
set(GLEW_DIR "c:/Users/qkrwh/3rdparty/install/glew/lib/cmake/glew")
find_package(GLEW REQUIRED)

if(GLEW_FOUND)
    message(STATUS "GLEW DIR : ${GLEW_DIR}")
    message(STATUS "GLEW INCLUDE DIR : ${GLEW_INCLUDE_DIRS}")
endif()

include_directories(${GLEW_INCLUDE_DIRS})


# Pangolin
set(Pangolin_DIR "c:/Users/qkrwh/3rdparty/install/Pangolin/lib/cmake/Pangolin")
find_package(Pangolin REQUIRED)

if(Pangolin_FOUND)
    #set(Pangolin_INCLUDE_DIRS "../../install/Pangolin/include")
    message(STATUS "Pangolin DIR : ${Pangolin_DIR}")
    #message(STATUS "Pangolin INCLUDE DIR : ${Pangolin_INCLUDE_DIRS}")
    message(STATUS "Pangolin LIBRARIES : ${Pangolin_LIBRARIES}")
endif()

# OpenCV
set(OpenCV_DIR "c:/Users/qkrwh/3rdparty/install/opencv")
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV_DIR : ${OpenCV_DIR}")
    message(STATUS "OpenCV_INCLUDE_DIRS : ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "OpenCV_LIBS : ${OpenCV_LIBS}")
endif()

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(joinMap ${OpenCV_LIBS} ${Pangolin_LIBRARIES})