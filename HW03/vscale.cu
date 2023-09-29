#include "vscale.cuh"

__global__ void vscale(const float *a, float *b, unsigned int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) {
        b[tid] = a[tid] * b[tid];
    }
}