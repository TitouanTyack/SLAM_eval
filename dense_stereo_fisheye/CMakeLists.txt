cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set (CMAKE_CXX_STANDARD 17)

project(pc_comparison)

# Find packages
find_package(PCL 1.2 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

# Include packages
include_directories(${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${YAML_CPP_INCLUDE_DIR})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Add libraries
add_library(denseStereo denseStereo.cpp denseStereo.hpp)

# Add executables
add_executable (tuning tuning.cpp)
target_link_libraries (tuning denseStereo ${PCL_LIBRARIES} ${OpenCV_LIBS} ${YAML_CPP_LIBRARIES} )

add_executable (pc_generation pc_generation)
target_link_libraries (pc_generation denseStereo ${PCL_LIBRARIES} ${OpenCV_LIBS} ${YAML_CPP_LIBRARIES} )
