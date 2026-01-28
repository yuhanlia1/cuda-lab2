// A tiling alg to compute matrix 

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

// kernel
__global__ void matrixMultiplyTiled(float *A, float *B, float *C, int N) {
	__shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
	__shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;

	int Row = by * TILE_WIDTH + ty; 
	int Col = bx * TILE_WIDTH + tx;

	float Pvalue = 0.0;
	for (int m = 0; m < (N + TILE_WIDTH - 1) / TILE_WIDTH; ++m) { 
 		if (Row < N && (m*TILE_WIDTH+tx) < N) 
 			ds_A[ty][tx] = A[Row * N + m * TILE_WIDTH + tx]; 
 		else 
 			ds_A[ty][tx] = 0.0f;
 		if (Col < N && (m*TILE_WIDTH+ty) < N) 
			ds_B[ty][tx] = B[(m*TILE_WIDTH + ty) * N + Col]; 
 		else 
 			ds_B[ty][tx] = 0.0f; 
 		__syncthreads(); 
 		for (int k = 0; k < TILE_WIDTH; ++k) 
 			Pvalue += ds_A[ty][k] * ds_B[k][tx]; 
 		__syncthreads(); 
 	}

	 if (Row < N && Col < N) 
 		C[Row * N + Col] = Pvalue; 

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
	dim3 block(TILE_WIDTH, TILE_WIDTH);
	dim3 grid((N + TILE_WIDTH - 1) / 16, (N + TILE_WIDTH - 1) / 16);	

	// running time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	matrixMultiplyTiled<<<grid, block>>>(d_A, d_B, d_C, N);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	
	cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

	// timing record
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	float gpu_seconds = milliseconds / 1000.0f;
	printf("GPU execution time (N=%d): %f seconds\n", N, gpu_seconds);
	
	cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
	
	return 0;
}
