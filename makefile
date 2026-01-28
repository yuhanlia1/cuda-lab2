# ========================
# Compiler settings
# ========================
CC      = gcc
CXX     = g++
NVCC    = nvcc

CFLAGS  = -O2
NVFLAGS = -O2

# ========================
# Directories
# ========================
BUILD_DIR = build
MATRIX_DIR = matrix
CONV_DIR   = conv
SHARED_DIR = shared_lib

# ========================
# Targets
# ========================
MATRIX_CPU     = $(BUILD_DIR)/matrix_cpu
MATRIX_NAIVE   = $(BUILD_DIR)/matrix_naive
MATRIX_OPT     = $(BUILD_DIR)/matrix_opt
MATRIX_CUBLAS  = $(BUILD_DIR)/matrix_cuBLAS

CONV_CPU       = $(BUILD_DIR)/conv_cpu
CONV_CUDA      = $(BUILD_DIR)/conv_cuda

SHARED_LIB = $(BUILD_DIR)/lib_matrix_conv.so
SHARED_TEST = $(BUILD_DIR)/test_shared.py

# ========================
# Default target
# ========================
all: $(BUILD_DIR) matrix conv shared

# ========================
# Create build directory
# ========================
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# ========================
# Matrix targets
# ========================
matrix: $(MATRIX_CPU) $(MATRIX_NAIVE) $(MATRIX_OPT) $(MATRIX_CUBLAS)

$(MATRIX_CPU): $(MATRIX_DIR)/matrix_cpu.c
	$(CC) $(CFLAGS) $< -o $@

$(MATRIX_NAIVE): $(MATRIX_DIR)/matrix_naive.cu
	$(NVCC) $(NVFLAGS) $< -o $@

$(MATRIX_OPT): $(MATRIX_DIR)/matrix_opt.cu
	$(NVCC) $(NVFLAGS) $< -o $@

$(MATRIX_CUBLAS): $(MATRIX_DIR)/matrix_cuBLAS.cu
	$(NVCC) $(NVFLAGS) $< -o $@ -lcublas

# ========================
# Convolution targets
# ========================
conv: $(CONV_CPU) $(CONV_CUDA)

$(CONV_CPU): $(CONV_DIR)/conv_cpu.c
	$(CC) $(CFLAGS) $< -o $@

$(CONV_CUDA): $(CONV_DIR)/conv_cuda.cu
	$(NVCC) $(NVFLAGS) $< -o $@

# ========================
# Shared lib targets
# ========================
shared: $(SHARED_LIB) $(SHARED_TEST)

$(SHARED_LIB): $(SHARED_DIR)/matrix_conv_lib.cu
	$(NVCC) -Xcompiler -fPIC -shared $< -o $@
	
$(SHARED_TEST): $(SHARED_DIR)/test_shared.py | $(BUILD_DIR)
	cp $< $@

# ========================
# Test targets
# ========================
test: all
	@echo "===== Matrix multiplication tests ====="
	@echo "--- N = 512 ---"
	$(MATRIX_CPU) 512
	$(MATRIX_NAIVE) 512
	$(MATRIX_OPT) 512
	$(MATRIX_CUBLAS) 512

	@echo "--- N = 1024 ---"
	$(MATRIX_CPU) 1024
	$(MATRIX_NAIVE) 1024
	$(MATRIX_OPT) 1024
	$(MATRIX_CUBLAS) 1024

	@echo "--- N = 2048 ---"
	$(MATRIX_CPU) 2048
	$(MATRIX_NAIVE) 2048
	$(MATRIX_OPT) 2048
	$(MATRIX_CUBLAS) 2048

	@echo "===== Convolution tests ====="
	@echo "--- M = 512, K = 3 ---"
	$(CONV_CPU) 512 3
	$(CONV_CUDA) 512 3

	@echo "--- M = 1024, K = 5 ---"
	$(CONV_CPU) 1024 5
	$(CONV_CUDA) 1024 5

	@echo "===== Shared library (Python) test ====="
	cd $(BUILD_DIR) && python3 test_shared.py

# ========================
# Clean
# ========================
clean:
	rm -rf $(BUILD_DIR)

.PHONY: all matrix conv test clean

