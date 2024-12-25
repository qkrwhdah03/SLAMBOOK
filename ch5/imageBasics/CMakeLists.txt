cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED on)

project(imageBasics)


# OpenCV
set(OpenCV_DIR "../../install/opencv/")
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV_DIR : ${OpenCV_DIR}")
    message(STATUS "OpenCV_INCLUDE_DIRS : ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "OpenCV_LIBS : ${OpenCV_LIBS}")
endif()
include_directories(${OpenCV_INCLUDE_DIRS})


# imageBasics.cpp
add_executable(imageBasics imageBasics.cpp)
target_link_libraries(imageBasics ${OpenCV_LIBS})



# undistorImage.cpp
#add_executable(undistortImage undistortImage.cpp)
#target_link_libraries(undistortImage ${OpenCV_LIBS})
