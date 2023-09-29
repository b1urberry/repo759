#include <stdio.h>

__global__ void factorialKernel(int *results) {
    int tid = threadIdx.x; // Get the current thread's ID.
    int factorial = 1;

    for (int i = 1; i <= tid + 1; i++) { // tid starts from 0, so add 1 to get the actual number.
        factorial *= i;
    }

    results[tid] = factorial;
}

int main() {
    const int N = 8;
    int host_results[N]; 
    int *device_results;

    // Allocate memory on the GPU.
    cudaMalloc(&device_results, N * sizeof(int));

    // Launch the kernel with 1 block and 8 threads.
    factorialKernel<<<1, N>>>(device_results);

    // Copy results from device to host.
    cudaMemcpy(host_results, device_results, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Synchronize device to ensure completion of the kernel before proceeding.
    cudaDeviceSynchronize();

    for (int i = 0; i < N; i++) {
        printf("%d!=%d\n", i+1, host_results[i]);
    }

    // Free GPU memory.
    cudaFree(device_results);

    return 0;
}
