// add_kernel.cu
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>   // at::cuda::getCurrentCUDAStream
#include <c10/cuda/CUDAGuard.h>      // c10::cuda::CUDAGuard

namespace {

__global__ void vector_add_kernel(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ c,
                                  int64_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

torch::Tensor vector_add_cuda(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(a.scalar_type() == torch::kFloat32,
                "a must be float32");
    TORCH_CHECK(b.scalar_type() == torch::kFloat32,
                "b must be float32");
    TORCH_CHECK(a.sizes() == b.sizes(),
                "a and b must have the same shape");

    a = a.contiguous();
    b = b.contiguous();

    auto c = torch::empty_like(a);
    int64_t n = a.numel();

    const int threads = 256;
    const int blocks  = static_cast<int>((n + threads - 1) / threads);

    // 保证当前 CUDA device 与张量所在 device 一致
    c10::cuda::CUDAGuard device_guard(a.device());

    // 拿到当前 stream（对应这个 device）
    auto stream = at::cuda::getCurrentCUDAStream(a.device().index());

    vector_add_kernel<<<blocks, threads, 0, stream>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        n
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "CUDA kernel launch failed");

    return c;
}

} // namespace

PYBIND11_MODULE(add_cuda, m) {
    m.def("add", &vector_add_cuda,
          "Vector addition (CUDA, takes torch.Tensor)");
}
