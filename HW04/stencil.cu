// Author: Jiayi Liu

// Computes the convolution of image and mask, storing the result in output.
// Each thread should compute _one_ element of the output matrix.
// Shared memory should be allocated _dynamically_ only.
//
// image is an array of length n.
// mask is an array of length (2 * R + 1).
// output is an array of length n.
// All of them are in device memory
//
// Assumptions:
// - 1D configuration
// - blockDim.x >= 2 * R + 1
//
// The following should be stored/computed in shared memory:
// - The entire mask
// - The elements of image that are needed to compute the elements of output corresponding to the threads in the given block
// - The output image elements corresponding to the given block before it is written back to global memory
__global__ void stencil_kernel(const float* image, const float* mask, float* output, unsigned int n, int R)
{
    extern __shared__ float sharedData[]; // dynamic shared memory declaration

    float *sharedImage = &sharedData[0];  // For image data
    float *sharedMask  = &sharedData[blockDim.x + 2*R]; // For mask data

    unsigned int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    // Load mask into shared memory
    if(tid < 2*R + 1) {
        sharedMask[tid] = mask[tid];
    }

    // Load image elements into shared memory
    sharedImage[tid + R] = (i < n) ? image[i] : 1;

    // Handle the halo elements
    if(tid < R) {
        sharedImage[tid] = (i - R >= 0) ? image[i - R] : 1;
        sharedImage[tid + blockDim.x + R] = (i + blockDim.x < n) ? image[i + blockDim.x] : 1;
    }

    __syncthreads();  // Ensure all data is loaded before computation starts

    float result = 0.0f;
    if(i < n) {
        for(int j = -R; j <= R; j++) {
            result += sharedImage[tid + j + R] * sharedMask[j + R];
        }
        output[i] = result;
    }
}

// Makes one call to stencil_kernel with threads_per_block threads per block.
// You can consider following the kernel call with cudaDeviceSynchronize (but if you use
// cudaEventSynchronize to time it, that call serves the same purpose as cudaDeviceSynchronize).
//
// Assumptions:
// - threads_per_block >= 2 * R + 1
__host__ void stencil(const float* image,
                      const float* mask,
                      float* output,
                      unsigned int n,
                      unsigned int R,
                      unsigned int threads_per_block)
{
    // Size of dynamic shared memory
    unsigned int sharedMemSize = (threads_per_block + 2*R) * sizeof(float) + (2*R + 1) * sizeof(float);

    // Launching the kernel
    unsigned int numBlocks = (n + threads_per_block - 1) / threads_per_block;
    stencil_kernel<<<numBlocks, threads_per_block, sharedMemSize>>>(image, mask, output, n, R);

}
