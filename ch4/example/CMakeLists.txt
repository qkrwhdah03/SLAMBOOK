cmake_minimum_required(VERSION 3.1)
project(trajectoryError)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_UBUNTU_20 "Set to ON if you are using Ubuntu 20.04" OFF)
if(USE_UBUNTU_20)
    message("You are using Ubuntu 20.04, fmt::fmt will be linked")
    find_package(fmt REQUIRED)
    set(FMT_LIBRARIES fmt::fmt)
endif()

# Executable
add_executable(trajectoryError trajectoryError.cpp)

# Eigen
target_include_directories(trajectoryError PRIVATE "c:/Users/qkrwh/3rdparty/install/eigen3/include")

# Threads 
find_package(Threads REQUIRED)
target_link_libraries(trajectoryError PRIVATE Threads::Threads)

# GLEW 
set(GLEW_DIR "c:/Users/qkrwh/3rdparty/install/glew/lib/cmake/glew")
find_package(GLEW REQUIRED)

if(GLEW_FOUND)
    message(STATUS "GLEW DIR : ${GLEW_DIR}")
    message(STATUS "GLEW INCLUDE DIR : ${GLEW_INCLUDE_DIR}")
    message(STATUS "GLEW LIBRARIES : ${GLEW_LIBRARIES}")
endif()

target_include_directories(trajectoryError PRIVATE ${GLEW_INCLUDE_DIRS})
target_link_libraries(trajectoryError PRIVATE ${GLEW_LIBRARIES})

# Pangolin
set(Pangolin_DIR "c:/Users/qkrwh/3rdparty/install/Pangolin/lib/cmake/Pangolin")
find_package(Pangolin REQUIRED)

if(Pangolin_FOUND)
    message(STATUS "Pangolin DIR : ${Pangolin_DIR}")
    message(STATUS "Pangolin INCLUDE DIR : ${Pangolin_INCLUDE_DIRS}")
    message(STATUS "Pangolin LIBRARIES : ${Pangolin_LIBRARIES}")
endif()

target_include_directories(trajectoryError PRIVATE ${Pangolin_INCLUDE_DIRS})
target_link_libraries(trajectoryError PRIVATE ${Pangolin_LIBRARIES} ${FMT_LIBRARIES})

# Sophus
set(Sophus_DIR "c:/Users/qkrwh/3rdparty/install/Sophus/share/sophus/cmake")
find_package(Sophus REQUIRED)

if(Sophus_FOUND)
    message(STATUS "Sophus_DIR : ${Sophus_DIR}")
    message(STATUS "Sophus_INCLUDE_DIR : ${Sophus_INCLUDE_DIRS}")
    message(STATUS "Sophus_LIBRARIES : ${Sophus_LIBRARIES}")
endif()

target_link_libraries(trajectoryError PRIVATE Sophus::Sophus)
target_include_directories(trajectoryError PRIVATE ${Sophus_INCLUDE_DIR})