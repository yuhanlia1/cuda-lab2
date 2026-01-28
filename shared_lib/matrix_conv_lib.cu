// cuda matrix mult 

#include <cuda_runtime.h> 
#include <stdio.h> 
#define TILE_WIDTH 16 

// Shared lib1: tiled matrix multiply 
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

// extern func
extern "C" void gpu_matrix_multiply(float *h_A, float *h_B, float *h_C, int N) { 
	size_t size = N * N * sizeof(float); 
	float *d_A, *d_B, *d_C; 
	
	cudaMalloc((void**)&d_A, size); 
	cudaMalloc((void**)&d_B, size); 
	cudaMalloc((void**)&d_C, size); 
	
	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice); 
	cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice); 
	
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH); 
	dim3 dimGrid((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH); 
	
	matrixMultiplyTiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N); 
	cudaDeviceSynchronize(); 
	
	cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost); 
	cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); 
}


// Shared lib2: convolution kernel (image*kernel=output)
__global__ void convolution2D_GPU(unsigned int *image, int *kernel, unsigned int *output, int M, int N
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int pad = N / 2;

    if (x >= pad && x < M - pad && y >= pad && y < M - pad) {
        int sum = 0;
        for (int ky = -pad; ky <= pad; ky++) {
            for (int kx = -pad; kx <= pad; kx++) {
                sum += image[(y + ky) * M + (x + kx)] *
                       kernel[(ky + pad) * N + (kx + pad)];
            }
        }
        output[y * M + x] = (unsigned int)sum;
    }
}

extern "C" void gpu_convolution(unsigned int *h_image, int *h_kernel, unsigned int *h_output, int M, int N) {
    size_t img_size = M * M * sizeof(unsigned int);
    size_t ker_size = N * N * sizeof(int);

    unsigned int *d_image, *d_output;
    int *d_kernel;

    cudaMalloc(&d_image, img_size);
    cudaMalloc(&d_output, img_size);
    cudaMalloc(&d_kernel, ker_size);

    cudaMemcpy(d_image, h_image, img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, ker_size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((M + 15) / 16, (M + 15) / 16);

    convolution2D_GPU<<<grid, block>>>(d_image, d_kernel, d_output, M, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, img_size, cudaMemcpyDeviceToHost);

    cudaFree(d_image);
    cudaFree(d_kernel);
    cudaFree(d_output);
}

