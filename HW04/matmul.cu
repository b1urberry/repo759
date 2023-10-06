#include "matmul.cuh"

#include <stdio.h>
#include <cmath>


// Computes the matrix product of A and B, storing the result in C.
// Each thread should compute _one_ element of output.
// Does not use shared memory for this problem.
//
// A, B, and C are row major representations of nxn matrices in device memory.
//
// Assumptions:
// - 1D kernel configuration
__global__ void matmul_kernel(const float* A, const float* B, float* C, size_t n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // get current thread's ID
    int row_idx = tid / n;
    int col_idx = tid % n;
    float sum = 0;
    
    for (int i = 0; i < n; i ++) 
    {
        sum += A[row_idx * n + i] * B[i * n + col_idx];
    }

    C[tid] = sum;
}

// Makes one call to matmul_kernel with threads_per_block threads per block.
// You can consider following the kernel call with cudaDeviceSynchronize (but if you use 
// cudaEventSynchronize to time it, that call serves the same purpose as cudaDeviceSynchronize).
void matmul(const float* A, const float* B, float* C, size_t n, unsigned int threads_per_block)
{
    int number_of_blocks = (n * n + threads_per_block - 1 ) / threads_per_block;
    matmul_kernel<<<number_of_blocks, threads_per_block>>>(A, B, C, n);
}