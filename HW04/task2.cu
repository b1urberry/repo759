#include <iostream>
#include <curand_kernel.h>
#include <cstdlib>
#include <random>
#include <chrono>

#include "stencil.cuh"

using namespace std;
using std::cout;
using std::endl;
using std::chrono::high_resolution_clock;
using std::chrono::duration;
using std::vector;

// Helper function to initialize data
void fillRandom(float* data, size_t size) {
    std::mt19937 rng(static_cast<unsigned int>(std::time(nullptr))); // seed the random number generator
    std::uniform_real_distribution<float> dist(-1.0, 1.0); // uniform distribution between -1.0 and 1.0

    for(size_t i = 0; i < size; i++) {
        data[i] = dist(rng); // random float between [-1, 1]
    }
}

int main(int argc, char** argv) {
    if (argc != 4) {
        return 1;
    }

    unsigned int n = atoi(argv[1]);
    unsigned int R = atoi(argv[2]);
    unsigned int threads_per_block = atoi(argv[3]);

    float *h_image = new float[n];
    float *h_output = new float[n];
    float *h_mask = new float[2 * R + 1];

    // Fill host arrays with random numbers
    fillRandom(h_image, n);
    fillRandom(h_mask, 2 * R + 1);

    float *d_image, *d_output, *d_mask;

    // Allocate device memory
    cudaMalloc(&d_image, n * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));
    cudaMalloc(&d_mask, (2 * R + 1) * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_image, h_image, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask, (2 * R + 1) * sizeof(float), cudaMemcpyHostToDevice);

    // Record the start time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Call the stencil function
    stencil(d_image, d_mask, d_output, n, R, threads_per_block);

    // Record the stop time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy results back to the host
    cudaMemcpy(h_output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the last element and the execution time
    printf("%.2f\n", h_output[n - 1]);
    printf("%.2f\n", milliseconds);

    // Clean up
    delete[] h_image;
    delete[] h_output;
    delete[] h_mask;

    cudaFree(d_image);
    cudaFree(d_output);
    cudaFree(d_mask);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
