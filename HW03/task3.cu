// task3.cu

#include <stdio.h>
#include <curand_kernel.h>
#include "vscale.cuh"

const int TPB = 16;  // Threads Per Block

int main(int argc, char *argv[]) {
    if (argc != 2) {
        return 1;
    }

    int n = atoi(argv[1]);

    float *a, *b, *d_a, *d_b;
    a = (float *)malloc(n * sizeof(float));
    b = (float *)malloc(n * sizeof(float));

    // Fill a and b with random numbers
    for (int i = 0; i < n; i++) {
        a[i] = ((float)rand() / RAND_MAX) * 20.0 - 10.0;  // Values between -10 and 10
        b[i] = (float)rand() / RAND_MAX;  // Values between 0 and 1
    }

    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));

    cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int numBlocks = (n + TPB - 1) / TPB;

    cudaEventRecord(start);
    vscale<<<numBlocks, TPB>>>(d_a, d_b, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(b, d_b, n * sizeof(float), cudaMemcpyDeviceToHost);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    printf("%f\n%f\n%f\n", ms, b[0], b[n-1]);

    free(a);
    free(b);
    cudaFree(d_a);
    cudaFree(d_b);

    return 0;
}
