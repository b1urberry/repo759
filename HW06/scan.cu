// hillis_steele kernel for scanning 
__global__ void hillis_steele(const float *input, float *output, int n) {
    extern volatile __shared__ float temp[];  // allocated on invocation
    int thid = threadIdx.x;
    int pout = 0, pin = 1;

    // Load input into shared memory.
    temp[pout*n + thid] = (thid > 0) ? input[thid-1] : 0;
    __syncthreads();

    for (int offset = 1; offset < n; offset *= 2) {
        pout = 1 - pout; // toggle double buffer indices
        pin = 1 - pout;
        if (thid >= offset) {
            temp[pout*n+thid] = temp[pin*n+thid - offset] + temp[pin*n+thid];
        } else {
            temp[pout*n+thid] = temp[pin*n+thid];
        }
        __syncthreads();
    }

    output[thid] = temp[pout*n+thid];  // write output
}


__host__ void scan(const float* input, float* output, unsigned int n, unsigned int threads_per_block) {
    // Calculate number of blocks required
    unsigned int num_blocks = (n + threads_per_block - 1) / threads_per_block;

    // Launch the kernel
    hillis_steele<<<num_blocks, threads_per_block, 2 * threads_per_block * sizeof(float)>>>(input, output, n);
}


