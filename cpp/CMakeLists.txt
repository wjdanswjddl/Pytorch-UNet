cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(test-ts)

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I /cvmfs/larsoft.opensciencegrid.org/products/eigen/v3_3_4a/include/eigen3/")
add_executable(test-ts thread.cpp)
target_link_libraries(test-ts "${TORCH_LIBRARIES}" -lpthread)
set_property(TARGET test-ts PROPERTY CXX_STANDARD 11)
