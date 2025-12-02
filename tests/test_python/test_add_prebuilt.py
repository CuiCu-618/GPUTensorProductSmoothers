# test_add_makefile.py
import torch
import add_cuda  # 来自 Makefile 编译出的 add_cuda*.so

def add_cuda_impl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and b.is_cuda
    assert a.dtype == torch.float32 and b.dtype == torch.float32
    assert a.shape == b.shape

    c = torch.empty_like(a)
    n = a.numel()

    # 传递 device pointer（Python 里的 int）
    add_cuda.add(
        int(a.data_ptr()),
        int(b.data_ptr()),
        int(c.data_ptr()),
        n,
    )

    # 如果担心异步，可以在这里同步一下
    torch.cuda.synchronize()
    return c

def add_torch_impl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a + b

def main():
    device = torch.device("cuda:0")

    n = 1024 * 1024
    a = torch.randn(n, device=device, dtype=torch.float32)
    b = torch.randn(n, device=device, dtype=torch.float32)

    c_cuda  = add_cuda_impl(a, b)
    c_torch = add_torch_impl(a, b)

    if torch.allclose(c_cuda, c_torch, atol=1e-6):
        print("✅ CUDA(add_cuda) matches PyTorch add!")
    else:
        max_diff = (c_cuda - c_torch).abs().max().item()
        print("❌ Mismatch, max diff:", max_diff)

if __name__ == "__main__":
    main()
