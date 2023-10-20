// task2.cu
#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <random>
#include <chrono>


#include "scan.cuh"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <n> <threads_per_block>" << std::endl;
        return 1;
    }

    int n = atoi(argv[1]);
    int threads_per_block = atoi(argv[2]);

    // Allocate managed memory
    float *input, *output;
    cudaMallocManaged(&input, n*sizeof(float));
    cudaMallocManaged(&output, n*sizeof(float));

    std::mt19937 rng(static_cast<unsigned int>(std::time(nullptr))); // seed the random number generator
    std::uniform_real_distribution<float> dist(-1.0, 1.0); 
    // Fill input with random float numbers in range [-1, 1]
    for (int i = 0; i < n; ++i) {
        input[i] = dist(rng);
    }

    // Time the scan operation using CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    scan(input, output, n, threads_per_block);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // Print the last element of the array and the time taken
    std::cout << output[n-1] << std::endl;
    std::cout << elapsedTime << std::endl;

    cudaFree(input);
    cudaFree(output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
