#include <pybind11/pybind11.h>

PYBIND11_MODULE(knn, m) {
    m.doc() = "Stub KNN extension (no-op)";
    m.def("version", []() { return "stub"; });
}
