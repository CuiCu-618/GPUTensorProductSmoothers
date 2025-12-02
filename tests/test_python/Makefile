# Makefile

# ========= 基本配置 =========
CUDA_HOME ?= /usr/local/cuda
NVCC      ?= $(CUDA_HOME)/bin/nvcc

PYTHON        ?= python3.9
PYTHON_CONFIG ?= python3-config

TARGET = add_cuda
SRC    = add_kernel.cu

# ========= Python / pybind11 路径 =========
PYTHON_INCLUDES := $(shell $(PYTHON_CONFIG) --includes)
PYTHON_LDFLAGS  := $(shell $(PYTHON_CONFIG) --ldflags)

PYBIND11_INCLUDES := $(shell $(PYTHON) -m pybind11 --includes)

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
	    $(PYTHON_INCLUDES) $(PYBIND11_INCLUDES) \
	    $(PYTHON_LDFLAGS) \
	    -o $@

clean:
	rm -f $(TARGET)*.so *.o *.cubin *.ptx *.linkinfo
