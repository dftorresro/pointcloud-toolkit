cmake_minimum_required(VERSION 3.20)
project(pcseg_kernels LANGUAGES CXX)


set(PYBIND11_FINDPYTHON ON CACHE BOOL "")


find_package(pybind11 CONFIG REQUIRED)


add_library(knn MODULE knn.cpp)
target_compile_features(knn PRIVATE cxx_std_17)

# proper module filename:  knn.cpython-310-darwin.so
set_target_properties(knn PROPERTIES
    PREFIX ""
    SUFFIX ".so")

# link against pybind11; this also propagates the include dirs
target_link_libraries(knn PRIVATE pybind11::module pybind11::pybind11)
