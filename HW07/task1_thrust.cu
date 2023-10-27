#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <random>
#include <chrono>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


int main(int argc, char** argv) {

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << "n" << std::endl;
        return 1;
    }

    int n = atoi(argv[1]);

    // create the host vector
    thrust::host_vector<float> H(n);
    
    // fill the host vector H with random float numbers in range [-1, 1]
    std::mt19937 rng(static_cast<unsigned int>(std::time(nullptr))); // seed the random number generator
    std::uniform_real_distribution<float> dist(-1.0, 1.0); 
    for (int i = 0; i < n; ++i) {
        H[i] = dist(rng);
    }

    thrust::device_vector<float> D = H;

    // Time the scan operation using CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    float result = thrust::reduce(D.begin(), D.end());
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // Print the last element of the array and the time taken
    std::cout << result << std::endl;
    std::cout << elapsedTime << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
