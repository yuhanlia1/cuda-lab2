#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// kernel
__global__ void conv2d_GPU(
    const unsigned int *image,
    const int *kernel,
    unsigned int *output,
    int M, int N
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

// host
int main(int argc, char **argv) {

    // Image size M x M, kernel size N x N
    int M = (argc > 1) ? atoi(argv[1]) : 512;
    int N = (argc > 2) ? atoi(argv[2]) : 3;

    size_t img_size = M * M * sizeof(unsigned int);
    size_t ker_size = N * N * sizeof(int);

    unsigned int *h_image  = (unsigned int *)malloc(img_size);
    unsigned int *h_output = (unsigned int *)malloc(img_size);
    int *h_kernel           = (int *)malloc(ker_size);

    // Initialize
    for (int i = 0; i < M * M; i++) {
        h_image[i] = rand() % 256;		// 8-bit grayscale
        h_output[i] = 0;
    }
	for (int i = 0; i < N * N; i++) {
		h_kernel[i] = (rand() % 3) - 1;  // -1, 0, 1
	}

    // device memory
    unsigned int *d_image, *d_output;
    int *d_kernel;

    cudaMalloc((void**)&d_image, img_size);
    cudaMalloc((void**)&d_output, img_size);
    cudaMalloc((void**)&d_kernel, ker_size);

    cudaMemcpy(d_image, h_image, img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, ker_size, cudaMemcpyHostToDevice);

    // threads setting 
    dim3 block(16, 16);
    dim3 grid((M + 15) / 16, (M + 15) / 16);

	// running time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

    conv2d_GPU<<<grid, block>>>(d_image, d_kernel, d_output, M, N);
	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

    cudaMemcpy(h_output, d_output, img_size, cudaMemcpyDeviceToHost);

	// timing record
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	float gpu_seconds = milliseconds / 1000.0f;
    printf("CUDA convolution finished (M=%d, N=%d) in %f seconds\n", M, N, gpu_seconds);

    cudaFree(d_image);
    cudaFree(d_kernel);
    cudaFree(d_output);
    free(h_image);
    free(h_kernel);
    free(h_output);

    return 0;
}
