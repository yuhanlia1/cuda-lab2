// office library cuBLAS 

#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


int main(int argc, char **argv){
	int N = (argc > 1) ? atoi(argv[1]) : 1024;	// default size 1024
	size_t size = N*N*sizeof(float);

	// malloc CPU memory (host)
	float *h_A = (float *)malloc(size);
	float *h_B = (float *)malloc(size);
	float *h_C = (float *)malloc(size);

	// cuBLAS parameters:
	float alpha = 1.0f;
	float beta  = 0.0f;
	cublasHandle_t handle;
	cublasCreate(&handle);

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


	// running time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N);
	// OP_N&OP_T is the parameters to transpose

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	
	cublasDestroy(handle);
	cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

	// timing record
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	float gpu_seconds = milliseconds / 1000.0f;
	printf("GPU (cuBLAS) execution time (N=%d): %f seconds\n", N, gpu_seconds);
	
	cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
	
	return 0;
}

