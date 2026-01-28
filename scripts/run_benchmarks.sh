#!/usr/bin/env bash

set -e

ROOT_DIR=$(pwd)
BUILD_DIR="$ROOT_DIR/build"
RESULTS_DIR="$ROOT_DIR/results"
LOG_DIR="$ROOT_DIR/logs"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

####################################
# Matrix multiplication benchmarks
####################################
MATRIX_SIZES=(256 512 1024 2048 4096)

echo "N,impl,time_sec" > "$RESULTS_DIR/matrix_results.csv"

for N in "${MATRIX_SIZES[@]}"; do
    echo "===== Matrix N=$N ====="

    for impl in matrix_cpu matrix_naive matrix_opt matrix_cuBLAS; do
        echo "Running $impl N=$N"

        LOG_FILE="$LOG_DIR/${impl}_N${N}.log"
        OUTPUT=$("$BUILD_DIR/$impl" "$N" | tee "$LOG_FILE")

        # extract 
        TIME=$(echo "$OUTPUT" | grep -Eo '[0-9]+\.[0-9]+' | tail -n 1)

        echo "$N,$impl,$TIME" >> "$RESULTS_DIR/matrix_results.csv"
    done
done

####################################
# Convolution benchmarks
####################################
IMAGE_SIZES=(256 512 1024 2048)
KERNEL_SIZES=(3 5 7)

echo "M,K,impl,time_sec" > "$RESULTS_DIR/conv_results.csv"

for M in "${IMAGE_SIZES[@]}"; do
    for K in "${KERNEL_SIZES[@]}"; do
        echo "===== Convolution M=$M K=$K ====="

        for impl in conv_cpu conv_cuda; do
            echo "Running $impl M=$M K=$K"

            LOG_FILE="$LOG_DIR/${impl}_M${M}_K${K}.log"
            OUTPUT=$("$BUILD_DIR/$impl" "$M" "$K" | tee "$LOG_FILE")

            TIME=$(echo "$OUTPUT" | grep -Eo '[0-9]+\.[0-9]+' | tail -n 1)

            echo "$M,$K,$impl,$TIME" >> "$RESULTS_DIR/conv_results.csv"
        done
    done
done

####################################
# Shared library (Python) benchmark
####################################
echo "impl,time_sec" > "$RESULTS_DIR/shared_lib_results.csv"

echo "===== Shared library Python test ====="
LOG_FILE="$LOG_DIR/shared_lib_python.log"

OUTPUT=$(cd "$BUILD_DIR" && python3 ../build/test_shared.py | tee "$LOG_FILE")

# extract
MATRIX_TIME=$(echo "$OUTPUT" | grep "matrix mult" | grep -Eo '[0-9]+\.[0-9]+')
CONV_TIME=$(echo "$OUTPUT" | grep "convolution" | grep -Eo '[0-9]+\.[0-9]+')

echo "python_matrix,$MATRIX_TIME" >> "$RESULTS_DIR/shared_lib_results.csv"
echo "python_convolution,$CONV_TIME" >> "$RESULTS_DIR/shared_lib_results.csv"

echo "All benchmarks completed."

