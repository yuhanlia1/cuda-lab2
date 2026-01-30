import numpy as np
from PIL import Image
import subprocess
import os

# =========================
# Config
# =========================
IMAGE_FILE = "input.jpg"
IMAGE_TXT = "image.txt"
KERNEL_X_TXT = "kernel_x.txt"
KERNEL_Y_TXT = "kernel_y.txt"
OUT_X_TXT = "out_x.txt"
OUT_Y_TXT = "out_y.txt"
OUT_IMG = "edges.png"
EXEC = "./conv_cpu"

# =========================
# Helper: write image txt
# =========================
img = Image.open(IMAGE_FILE).convert("L")
img_arr = np.array(img, dtype=np.uint32)
M = img_arr.shape[0]

with open(IMAGE_TXT, "w") as f:
    f.write(f"{M}\n")
    for row in img_arr:
        f.write(" ".join(map(str, row)) + "\n")

print("Image written to image.txt")

# =========================
# Sobel kernels
# =========================
sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.int32)

sobel_y = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
], dtype=np.int32)

def write_kernel(kernel, filename):
    N = kernel.shape[0]
    with open(filename, "w") as f:
        f.write(f"{N}\n")
        for row in kernel:
            f.write(" ".join(map(str, row)) + "\n")

write_kernel(sobel_x, KERNEL_X_TXT)
write_kernel(sobel_y, KERNEL_Y_TXT)

print("Sobel kernels written")

# =========================
# Run C convolution
# =========================
subprocess.run([EXEC, IMAGE_TXT, KERNEL_X_TXT, OUT_X_TXT], check=True)
subprocess.run([EXEC, IMAGE_TXT, KERNEL_Y_TXT, OUT_Y_TXT], check=True)

print("C convolution finished for X and Y")

# =========================
# Read output txt
# =========================
def read_output(txt_file):
    with open(txt_file, "r") as f:
        lines = f.readlines()
    M = int(lines[0].strip())
    data = []
    for line in lines[1:]:
        data.append(list(map(float, line.split())))
    return np.array(data, dtype=np.float32)

Gx = read_output(OUT_X_TXT)
Gy = read_output(OUT_Y_TXT)

# =========================
# Gradient magnitude
# =========================
G = np.sqrt(Gx**2 + Gy**2)

# Normalize
G = G - G.min()
G = G / G.max()

# Threshold (key!)
threshold = 0.25
edges = (G > threshold).astype(np.uint8) * 255

# =========================
# Save result
# =========================
Image.fromarray(edges).save(OUT_IMG)
print(f"Final edge map saved to {OUT_IMG}")
