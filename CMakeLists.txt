cmake_minimum_required(VERSION 2.8.7)
project(STBA)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    message(STATUS "OpenMP is found")
else ()
    message(STATUS "OpenMP is not found")
endif()

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -std=c++11")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB SOURCES "src/*.cpp")
file(GLOB SUB_SOURCES "src/clustering/*.cpp")
set(SOURCES ${SOURCES} ${SUB_SOURCES})


add_executable(STBA ${SOURCES})
target_compile_definitions(STBA PUBLIC STBA_VERSION="0.0.0")
