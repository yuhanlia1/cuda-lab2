#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// conv func
void conv2d_CPU(unsigned int *image, int *kernel,
                unsigned int *output, int M, int N) {
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

// Read：
unsigned int *read_image(const char *filename, int *M) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        perror("Failed to open image file");
        exit(1);
    }

    fscanf(f, "%d", M);
    unsigned int *img = (unsigned int *)malloc((*M) * (*M) * sizeof(unsigned int));

    for (int i = 0; i < (*M) * (*M); i++) {
        fscanf(f, "%u", &img[i]);
    }

    fclose(f);
    return img;
}
int *read_kernel(const char *filename, int *N) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        perror("Failed to open kernel file");
        exit(1);
    }

    fscanf(f, "%d", N);
    int *ker = (int *)malloc((*N) * (*N) * sizeof(int));

    for (int i = 0; i < (*N) * (*N); i++) {
        fscanf(f, "%d", &ker[i]);
    }

    fclose(f);
    return ker;
}

// Write output to txt
void write_output(const char *filename, unsigned int *output, int M) {
    FILE *f = fopen(filename, "w");
    if (!f) {
        perror("Failed to open output file");
        exit(1);
    }

    fprintf(f, "%d\n", M);
    for (int i = 0; i < M * M; i++) {
        fprintf(f, "%u ", output[i]);
        if ((i + 1) % M == 0) 
			fprintf(f, "\n");
    }

    fclose(f);
}

int main(int argc, char **argv) {

    if (argc < 4) {
        printf("Usage: %s image.txt kernel.txt output.txt\n", argv[0]);
        return 1;
    }

    int M, N;
    unsigned int *image = read_image(argv[1], &M);
    int *kernel = read_kernel(argv[2], &N);

    if (N % 2 == 0) {
        printf("Kernel size must be odd\n");
        return 1;
    }

    unsigned int *output = (unsigned int *)calloc(M * M, sizeof(unsigned int));

    clock_t start = clock();
    conv2d_CPU(image, kernel, output, M, N);
    clock_t end = clock();

    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    //printf("CPU convolution finished (M=%d, N=%d) in %f seconds\n", M, N, elapsed);

    write_output(argv[3], output, M);

    free(image);
    free(kernel);
    free(output);
    return 0;
}

