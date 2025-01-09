cmake_minimum_required(VERSION 3.1)
project(useSophus)

# Set C++ 17 
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Executable
add_executable(useSophus useSophus.cpp)

# Sophus
set(Sophus_DIR "c:/Users/qkrwh/3rdparty/install/Sophus/share/sophus/cmake")
find_package(Sophus REQUIRED)

if(Sophus_FOUND)
    message(STATUS "Sophus_DIR : ${Sophus_DIR}")
    message(STATUS "Sophus_INCLUDE_DIR : ${Sophus_INCLUDE_DIRS}")
    message(STATUS "Sophus_LIBRARIES : ${Sophus_LIBRARIES}")
endif()

target_link_libraries(useSophus PUBLIC Sophus::Sophus)
target_include_directories(useSophus PRIVATE ${Sophus_INCLUDE_DIR})

# Eigen
target_include_directories(useSophus PRIVATE "c:/Users/qkrwh/3rdparty/install/eigen3/include")

# Example
add_subdirectory(example)
