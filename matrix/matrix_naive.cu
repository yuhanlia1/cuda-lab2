// A native cuda matrix computation

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// kernel
__global__ void matrixMultiplyGPU(float *A, float *B, float *C, int N) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
 	int col = blockIdx.x * blockDim.x + threadIdx.x;
 	if (row < N && col < N) {
 		float sum = 0.0f;
 		for (int k = 0; k < N; k++) {
 			sum += A[row * N + k] * B[k * N + col];
 		}
 		C[row * N + col] = sum;
 	}
}

// host
int main(int argc, char **argv){
	int N = (argc > 1) ? atoi(argv[1]) : 1024;	// default size 1024
	size_t size = N*N*sizeof(float);

	// malloc CPU memory (host)
	float *h_A = (float *)malloc(size);
	float *h_B = (float *)malloc(size);
	float *h_C = (float *)malloc(size);

	// initial 
	for (int i = 0; i < N*N; i++){
		h_A[i] = rand() % 100/100.0f;
		h_B[i] = rand() % 100/100.0f;
	}
	
	// GPU memory(device)
	float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

	// CPU -> GPU communicate 
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

	// threads setting:
	dim3 block(16, 16);
	dim3 grid((N+15) / 16, (N+15) / 16);	// ! ensure the grid should be greater than the elements of Matrix computation

	// running time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	matrixMultiplyGPU<<<grid, block>>>(d_A, d_B, d_C, N);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	
	cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

	// timing record
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	float gpu_seconds = milliseconds / 1000.0f;
	printf("GPU (naive) execution time (N=%d): %f seconds\n", N, gpu_seconds);
	
	cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
	
	return 0;
}

