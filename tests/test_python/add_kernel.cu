// add_kernel.cu
#include <pybind11/pybind11.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace py = pybind11;

__global__ void vector_add_kernel(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ c,
                                  int64_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// 这里不再用 torch::Tensor，而是用设备指针 + 长度
void vector_add_cuda(std::uintptr_t a_ptr,
                     std::uintptr_t b_ptr,
                     std::uintptr_t c_ptr,
                     std::int64_t n) {
    const float* a = reinterpret_cast<const float*>(a_ptr);
    const float* b = reinterpret_cast<const float*>(b_ptr);
    float*       c = reinterpret_cast<float*>(c_ptr);

    const int threads = 256;
    const int blocks  = static_cast<int>((n + threads - 1) / threads);

    vector_add_kernel<<<blocks, threads>>>(a, b, c, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA kernel launch failed: ") +
            cudaGetErrorString(err));
    }
}

PYBIND11_MODULE(add_cuda, m) {
    m.def("add", &vector_add_cuda,
          "Vector addition (CUDA, takes device pointers)",
          py::arg("a_ptr"),
          py::arg("b_ptr"),
          py::arg("c_ptr"),
          py::arg("n"));
}
