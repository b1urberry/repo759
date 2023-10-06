#include <curand_kernel.h>
#include <iostream>
#include <cstdlib>
#include <random>
#include <chrono>


#include "matmul.cuh"

using namespace std;
using std::cout;
using std::endl;
using std::chrono::high_resolution_clock;
using std::chrono::duration;
using std::vector;


int main(int argc, char* argv[]) 
{
    if (argc != 3) {
        return 1;
    }

    int n = atoi(argv[1]);
    int threads_per_block = atoi(argv[2]);

    // Create matrices A and B on host
    float *A = new float[n*n];
    float *B = new float[n*n];
    float *C = new float[n*n];

    // / filling the A and B matrices with random floats    
    std::mt19937 rng(static_cast<unsigned int>(std::time(nullptr))); // seed the random number generator
    std::uniform_real_distribution<float> dist1(-1.0, 1.0); // uniform distribution between -1.0 and 1.0
    std::uniform_real_distribution<float> dist2(-1.0, 1.0); // uniform distribution between -1.0 and 1.0

    // fill up A with random numbers between -1 and 1
    for (size_t i = 0; i < n*n; i++)
    {
        A[i] = dist1(rng);
        B[i] = dist2(rng);
    }

    float *d_A, *d_B, *d_C; // device A, B, C
    cudaMalloc(&d_A, n * n * sizeof(float));
    cudaMalloc(&d_B, n * n * sizeof(float));
    cudaMalloc(&d_C, n * n * sizeof(float));

    cudaMemcpy(d_A, A, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n * n * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul(d_A, d_B, d_C, n, threads_per_block);
    cudaEventRecord(stop);

    cudaMemcpy(C, d_C, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("%d %.2f %.2f\n", n, C[n * n - 1], milliseconds);

    delete[] A;
    delete[] B;
    delete[] C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}