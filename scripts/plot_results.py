import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import os

# =========================
# Paths
# =========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
FIG_DIR = os.path.join(ROOT_DIR, "figures")

os.makedirs(FIG_DIR, exist_ok=True)

print(f"Reading results from: {RESULTS_DIR}")
print(f"Saving figures to: {FIG_DIR}")

# =========================
# 1. Matrix multiplication plot
# =========================
matrix_csv = os.path.join(RESULTS_DIR, "matrix_results.csv")
df_mat = pd.read_csv(matrix_csv)

print("\nMatrix results preview:")
print(df_mat.head(10))

df_mat['N'] = df_mat['N'].astype(int)
df_mat['time_sec'] = df_mat['time_sec'].astype(float)

# follow the bash scripts order 
impl_order = ['matrix_cpu', 'matrix_naive', 'matrix_opt', 'matrix_cuBLAS']
impl_labels = {
    'matrix_cpu': 'CPU',
    'matrix_naive': 'Naïve CUDA',
    'matrix_opt': 'Optimized CUDA (Tiling)',
    'matrix_cuBLAS': 'cuBLAS'
}

plt.figure(figsize=(10, 6))
for impl in impl_order:
    sub = df_mat[df_mat["impl"] == impl].sort_values('N')
    if not sub.empty:
        plt.plot(sub["N"], sub["time_sec"], marker="o", 
                linewidth=2, markersize=8, label=impl_labels[impl])

plt.xlabel("Matrix size N", fontsize=12)
plt.ylabel("Execution time (seconds)", fontsize=12)
plt.title("Matrix Multiplication Performance Comparison", fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.yscale('log')  # because the cpu scale is huge 
plt.tight_layout()

output_path = os.path.join(FIG_DIR, "matrix_performance.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

if os.path.exists(output_path):
    print(f"✓ Matrix plot saved: {output_path}")
else:
    print(f"✗ Failed to save matrix plot")

# =========================
# 2. Convolution plot
# =========================
conv_csv = os.path.join(RESULTS_DIR, "conv_results.csv")
df_conv = pd.read_csv(conv_csv)

print("\nConvolution results preview:")
print(df_conv.head(10))

df_conv['M'] = df_conv['M'].astype(int)
df_conv['K'] = df_conv['K'].astype(int)
df_conv['time_sec'] = df_conv['time_sec'].astype(float)

df_conv['MK'] = df_conv.apply(
    lambda row: f"M={row['M']},K={row['K']}", axis=1
)

# (M,K)
df_conv = df_conv.sort_values(['M', 'K'])

x_labels = df_conv['MK'].unique()

x_positions = range(len(x_labels))

plt.figure(figsize=(12, 6))

impl_order_conv = ['conv_cpu', 'conv_cuda']
impl_labels_conv = {
    'conv_cpu': 'CPU Convolution',
    'conv_cuda': 'CUDA Convolution'
}

for impl in impl_order_conv:
    sub = df_conv[df_conv["impl"] == impl]
    # 按照x_labels的顺序获取时间
    times = []
    for label in x_labels:
        m, k = label.split(',')
        m_val = int(m.split('=')[1])
        k_val = int(k.split('=')[1])
        time_val = sub[(sub['M'] == m_val) & (sub['K'] == k_val)]['time_sec'].values
        if len(time_val) > 0:
            times.append(time_val[0])
        else:
            times.append(None)
    
    plt.plot(x_positions, times, marker="o", linewidth=2, 
            markersize=8, label=impl_labels_conv[impl])

plt.xlabel("Image size (M) and Kernel size (K)", fontsize=12)
plt.ylabel("Execution time (seconds)", fontsize=12)
plt.title("2D Convolution Performance Comparison", fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks(x_positions, x_labels, rotation=45, ha='right')
plt.tight_layout()

output_path = os.path.join(FIG_DIR, "convolution_performance.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

if os.path.exists(output_path):
    print(f"✓ Convolution plot saved: {output_path}")
else:
    print(f"✗ Failed to save convolution plot")

# =========================
# 3. Speedup analysis 
# =========================
plt.figure(figsize=(10, 6))

cpu_data = df_mat[df_mat["impl"] == "matrix_cpu"].sort_values('N')
for impl in ['matrix_naive', 'matrix_opt', 'matrix_cuBLAS']:
    impl_data = df_mat[df_mat["impl"] == impl].sort_values('N')
    if not impl_data.empty and not cpu_data.empty:
        speedup = []
        sizes = []
        for n in impl_data['N']:
            cpu_time = cpu_data[cpu_data['N'] == n]['time_sec'].values
            gpu_time = impl_data[impl_data['N'] == n]['time_sec'].values
            if len(cpu_time) > 0 and len(gpu_time) > 0:
                speedup.append(cpu_time[0] / gpu_time[0])
                sizes.append(n)
        
        if speedup:
            plt.plot(sizes, speedup, marker="o", linewidth=2, 
                    markersize=8, label=impl_labels[impl])

plt.xlabel("Matrix size N", fontsize=12)
plt.ylabel("Speedup vs CPU", fontsize=12)
plt.title("GPU Speedup over CPU (Matrix Multiplication)", fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Baseline (CPU)')
plt.tight_layout()

output_path = os.path.join(FIG_DIR, "matrix_speedup.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

if os.path.exists(output_path):
    print(f"✓ Speedup plot saved: {output_path}")

print("\n✓ All plots generated successfully!")