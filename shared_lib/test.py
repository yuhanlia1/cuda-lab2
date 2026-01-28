import ctypes
import numpy as np
import time
import os

# save the current path
lib_path = os.path.join(os.path.dirname(__file__), "lib_matrix_conv.so")

# Load shared library
lib = ctypes.cdll.LoadLibrary(lib_path)

# Matrix Mult
lib.gpu_matrix_multiply.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int
]

# Conv 
lib.gpu_convolution.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    ctypes.c_int
]


# Matrix size
S = 1024
A = np.random.rand(S, S).astype(np.float32)
B = np.random.rand(S, S).astype(np.float32)
C = np.zeros((S, S), dtype=np.float32)

# Conv parameters
M = 512
N = 3
image = np.random.randint(0, 256, size=(M, M), dtype=np.uint32)
kernel = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.int32)
# use the sobel kernel as example
output = np.zeros((M, M), dtype=np.uint32)

# Warm-up (important!)
lib.gpu_matrix_multiply(A.ravel(), B.ravel(), C.ravel(), S)
lib.gpu_convolution(image.ravel(), kernel.ravel(), output.ravel(), M, N)

# Timing
start1 = time.time()
lib.gpu_matrix_multiply(A.ravel(), B.ravel(), C.ravel(), S)
end1 = time.time()

start2 = time.time()
lib.gpu_convolution(image.ravel(), kernel.ravel(), output.ravel(), M, N)
end2 = time.time()

elapsed1 = end1 - start1;
elapsed2 = end2 - start2;

print(f"Python call to CUDA library completed the matrix mult in {elapsed1:.4f} seconds")
print(f"Python call to CUDA library completed the convolution in {elapsed2:.4f} seconds")

