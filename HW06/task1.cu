#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <cstdlib>
#include <random>
#include <chrono>

#include "mmul.h"

int main(int argc, char *argv[]) {
    // ensure the correct number of command-line arguments
    if (argc != 3) {
        printf("usage: ./task1 n n tests");
        return 1;
    }

    int n = atoi(argv[1]);
    int n_test = atoi(argv[2]);

    // Create matrices A, B, and C in managed memory
    float* A; cudaMallocManaged(&A, n * n * sizeof(float));
    float* B; cudaMallocManaged(&B, n * n * sizeof(float));
    float* C; cudaMallocManaged(&C, n * n * sizeof(float));

    // / filling the A and B matrices with random floats    
    std::mt19937 rng(static_cast<unsigned int>(std::time(nullptr))); // seed the random number generator
    std::uniform_real_distribution<float> dist1(-1.0, 1.0); // uniform distribution between -1.0 and 1.0
    std::uniform_real_distribution<float> dist2(-1.0, 1.0); // uniform distribution between -1.0 and 1.0
    std::uniform_real_distribution<float> dist3(-1.0, 1.0); // uniform distribution between -1.0 and 1.0

    // fill up A with random numbers between -1 and 1
    for (int i = 0; i < n*n; i++) {
        A[i] = dist1(rng);
        B[i] = dist2(rng);
        C[i] = dist3(rng);
    }

    float avg_time_ms = 0;

    for (int i = 0; i < n_test; i ++) {
        // define and create handle to this cuBLAS context
        cublasHandle_t handle;
        cublasCreate(&handle);
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        mmul(handle, A, B, C, n);
        cudaEventRecord(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        avg_time_ms += milliseconds;
    }

    avg_time_ms /= (float)n_test; 
    printf("%.2f\n", avg_time_ms);

    return 0;
}



