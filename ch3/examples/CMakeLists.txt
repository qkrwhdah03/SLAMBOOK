cmake_minimum_required(VERSION 2.8)

project(example)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# coordinateTransform.cpp
add_executable(coordinateTransform coordinateTransform.cpp)

target_include_directories(coordinateTransform PUBLIC "../../install/eigen3/include/eigen3")


# plotTrajectory.cpp
add_executable(plotTrajectory plotTrajectory.cpp)

# Eigen
target_include_directories(plotTrajectory PUBLIC "../../install/eigen3/include/eigen3")

# Threads
find_package(Threads REQUIRED)
target_link_libraries(plotTrajectory PUBLIC Threads::Threads)

# GLEW
set(GLEW_DIR "../../install/glew/lib/cmake/glew")
find_package(GLEW REQUIRED)

if(GLEW_FOUND)
    message(STATUS "GLEW DIR : ${GLEW_DIR}")
    message(STATUS "GLEW INCLUDE DIR : ${GLEW_INCLUDE_DIR}")
    message(STATUS "GLEW LIBRARIES : ${GLEW_LIBRARIES}")
endif()

target_include_directories(plotTrajectory PUBLIC ${GLEW_INCLUDE_DIRS})
target_link_libraries(plotTrajectory PUBLIC ${GLEW_LIBRARIES})

# Pangolin
set(Pangolin_DIR "../../install/Pangolin/lib/cmake/Pangolin")
find_package(Pangolin REQUIRED)
if(Pangolin_FOUND)
    set(Pangolin_INCLUDE_DIRS "../../install/Pangolin/include")
    message(STATUS "Pangolin DIR : ${Pangolin_DIR}")
    message(STATUS "Pangolin INCLUDE DIR : ${Pangolin_INCLUDE_DIRS}")
    message(STATUS "Pangolin LIBRARIES : ${Pangolin_LIBRARIES}")
endif()

target_include_directories(plotTrajectory PUBLIC ${Pangolin_INCLUDE_DIRS})
target_link_libraries(plotTrajectory pango_display pango_python)