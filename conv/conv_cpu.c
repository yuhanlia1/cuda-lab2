#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// CPU 2D convolution
void conv2d_CPU(unsigned int *image, int *kernel, unsigned int *output, int M, int N) {
    int pad = N / 2;

    for (int y = pad; y < M - pad; y++) {
        for (int x = pad; x < M - pad; x++) {
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
}

int main(int argc, char **argv) {

    // Image size and kernel size
    int M = (argc > 1) ? atoi(argv[1]) : 512;
    int N = (argc > 2) ? atoi(argv[2]) : 3;

    if (N % 2 == 0) {
        printf("Kernel size N must be odd\n");
        return -1;
    }

    size_t img_size = M * M * sizeof(unsigned int);
    size_t ker_size = N * N * sizeof(int);

    // Host memory
    unsigned int *image  = (unsigned int *)malloc(img_size);
    unsigned int *output = (unsigned int *)malloc(img_size);
    int *kernel = (int *)malloc(ker_size);

    // Initialize image
    for (int i = 0; i < M * M; i++) {
        image[i] = rand() % 256;   // grayscale
        output[i] = 0;
    }
	for (int i = 0; i < N * N; i++) {
		kernel[i] = (rand() % 3) - 1;  // -1, 0, 1
	}

    // Timing
    clock_t start = clock();
    conv2d_CPU(image, kernel, output, M, N);
    clock_t end = clock();

    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("CPU convolution finished (M=%d, N=%d) in %f seconds\n", M, N, elapsed);

    free(image);
    free(kernel);
    free(output);

    return 0;
}

