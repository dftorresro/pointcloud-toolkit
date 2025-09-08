cmake_minimum_required(VERSION 3.20)
project(pcseg_kernels LANGUAGES CXX)

# Make pybind11 respect the active Python from scikit-build/your env
set(PYBIND11_FINDPYTHON ON CACHE BOOL "")

find_package(pybind11 CONFIG REQUIRED)

# This sets up the right compile flags and extension naming automatically
pybind11_add_module(knn knn.cpp)
target_compile_features(knn PRIVATE cxx_std_17)

# Install the built extension at the top level of the wheel/site-packages
# (so `import knn` works). Change DESTINATION to "pcseg" if you want namespacing.
install(TARGETS knn
        LIBRARY DESTINATION .       # Unix/macOS
        RUNTIME DESTINATION .)      # Windows (pybind11 handles details)
