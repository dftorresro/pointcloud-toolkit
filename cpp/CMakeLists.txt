cmake_minimum_required(VERSION 3.20)
project(pcseg_kernels LANGUAGES CXX)

find_package(pybind11 CONFIG REQUIRED)

add_library(knn MODULE knn.cpp)
target_compile_features(knn PRIVATE cxx_std_17)
# produce proper Python extension name
set_target_properties(knn PROPERTIES PREFIX "" SUFFIX ".so")
target_link_libraries(knn PRIVATE pybind11::module)
