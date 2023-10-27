// task1_cub.cu

#include <stdio.h>
#include <stdlib.h>
#include <cub/cub.cuh>
#include <curand_kernel.h>
#include <random>
#include <chrono>

using namespace cub;

CachingDeviceAllocator g_allocator(true); // Caching allocator for device memory


int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Usage: %s <n>\n", argv[0]);
        exit(1);
    }

    int n = atoi(argv[1]);

    // Set up host arrays
    float *h_in = (float*)malloc(sizeof(float) * n);
    
    // fill the host vector H with random float numbers in range [-1, 1]
    std::mt19937 rng(static_cast<unsigned int>(std::time(nullptr))); // seed the random number generator
    std::uniform_real_distribution<float> dist(-1.0, 1.0); 
    for (int i = 0; i < n; ++i) {
        h_in[i] = dist(rng);
    }

    // Set up device arrays
    float *d_in = NULL;
    g_allocator.DeviceAllocate((void**)&d_in, sizeof(float) * n);
    cudaMemcpy(d_in, h_in, sizeof(float) * n, cudaMemcpyHostToDevice);

    // Setup device output array
    float *d_sum = NULL;
    g_allocator.DeviceAllocate((void**)&d_sum, sizeof(float));

    // Request and allocate temporary storage
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_sum, n);
    g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes);

    // Time the scan operation using CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_sum, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    float gpu_sum;
    cudaMemcpy(&gpu_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

    // Print the results
    std::cout << gpu_sum << std::endl;
    std::cout << milliseconds << std::endl;

    // Cleanup
    free(h_in);
    if (d_in) g_allocator.DeviceFree(d_in);
    if (d_sum) g_allocator.DeviceFree(d_sum);
    if (d_temp_storage) g_allocator.DeviceFree(d_temp_storage);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
