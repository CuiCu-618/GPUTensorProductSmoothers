# test_add_makefile.py
import torch
import add_cuda  # 来自 Makefile 编译出的 add_cuda*.so

def add_cuda_impl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return add_cuda.add(a, b)

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
