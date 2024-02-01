cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set (CMAKE_CXX_STANDARD 17)

project(pc_comparison)

find_package(PCL 1.2 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

include_directories(${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${YAML_CPP_INCLUDE_DIR})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


add_executable (pc_comparison pc_comparison.cpp)
target_link_libraries (pc_comparison ${PCL_LIBRARIES} ${OpenCV_LIBS} ${YAML_CPP_LIBRARIES} )