#include <stdio.h>
#include <curand_kernel.h>

const int ARRAY_SIZE = 16;
const int BLOCK_SIZE = 8;

__global__ void computeArray(int *dA, int a) {
    int tid = threadIdx.x;  // Get the current thread's ID within its block.
    int bid = blockIdx.x;  // Get the current block's ID.
    
    int idx = bid * BLOCK_SIZE + tid;
    
    int x = tid;
    int y = bid;

    dA[idx] = a * x + y;
}

int main() {
    int host_array[ARRAY_SIZE]; 
    int *dA;

    // Generate a random integer for 'a'
    a = rand() % 10; // Generate a random number between 0 and 9 for simplicity.

    // allocate an array of 16 ints on the device called dA from the host
    cudaMalloc(&dA, ARRAY_SIZE * sizeof(int));

    // Launch the kernel with 2 blocks and 8 threads per block.
    computeArray<<<2, BLOCK_SIZE>>>(dA, a);

    // Copy results from device to host.
    cudaMemcpy(host_array, dA, ARRAY_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the results.
    for (int i = 0; i < ARRAY_SIZE; i++) {
        printf("%d ", host_array[i]);
    }
    printf("\n");

    // Free GPU memory.
    cudaFree(dA);

    return 0;
}
