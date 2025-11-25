# Makefile

# ========= 基本配置 =========
CUDA_HOME ?= /usr/local/cuda
NVCC      ?= $(CUDA_HOME)/bin/nvcc

PYTHON        ?= python3
PYTHON_CONFIG ?= python3-config

TARGET = add_cuda
SRC    = add_kernel.cu

# ========= Python / PyTorch 路径 =========
PYTHON_INCLUDES := $(shell $(PYTHON_CONFIG) --includes)
PYTHON_LDFLAGS  := $(shell $(PYTHON_CONFIG) --ldflags)

TORCH_INCLUDES := $(shell $(PYTHON) -c "import torch; from torch.utils.cpp_extension import include_paths; print(' '.join('-I'+p for p in include_paths()))")
TORCH_LDFLAGS  := $(shell $(PYTHON) -c "import torch; from torch.utils.cpp_extension import library_paths; print(' '.join('-L'+p for p in library_paths()))")

# Python 扩展后缀，例如 .cpython-312-x86_64-linux-gnu.so
PY_EXT_SUFFIX := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

# ========= 编译选项 =========
CXX_STD    = c++17
NVCCFLAGS  = -O3 -std=$(CXX_STD) -Xcompiler -fPIC -shared
# A100 -> sm_80
NVCCFLAGS += -gencode arch=compute_80,code=sm_80

# ========= 目标 =========
all: $(TARGET)$(PY_EXT_SUFFIX)

$(TARGET)$(PY_EXT_SUFFIX): $(SRC)
	$(NVCC) $(SRC) $(NVCCFLAGS) \
	    $(PYTHON_INCLUDES) $(TORCH_INCLUDES) \
	    $(PYTHON_LDFLAGS) $(TORCH_LDFLAGS) \
	    -ltorch -ltorch_cpu -ltorch_python -lc10 \
	    -o $@

clean:
	rm -f $(TARGET)*.so *.o *.cubin *.ptx *.linkinfo
