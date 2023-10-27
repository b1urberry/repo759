// task2.cu

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/random.h>
#include <iostream>
#include "count.cuh" // assuming the header for the count function is named as such

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <n>" << std::endl;
        return 1;
    }

    const size_t n = std::stoi(argv[1]);
    
    thrust::host_vector<int> h_vec(n);
    thrust::random::default_random_engine rng;
    thrust::random::uniform_int_distribution<int> dist(0, 500);
    for (size_t i = 0; i < n; ++i) {
        h_vec[i] = dist(rng);
    }

    thrust::device_vector<int> d_vec = h_vec;
    thrust::device_vector<int> values;
    thrust::device_vector<int> counts;

    // Set up CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Time the count function
    cudaEventRecord(start);
    count(d_vec, values, counts);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << values.back() << std::endl;
    std::cout << counts.back() << std::endl;
    std::cout << milliseconds << std::endl;

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
