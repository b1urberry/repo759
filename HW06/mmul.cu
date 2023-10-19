#include "mmul.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

// Uses a single cuBLAS call to perform the operation C := A B + C
// handle is a handle to an open cuBLAS instance
// A, B, and C are matrices with n rows and n columns stored in column-major
// NOTE: The cuBLAS call should be followed by a call to cudaDeviceSynchronize() for timing purposes
void mmul(cublasHandle_t handle, const float* A, const float* B, float* C, int n) 
{
    // set constants alpha and beta to be 1
    float alpha = 1.0f;
    float beta = 1.0f;

    // Perform the matrix multiplication C := alpha * A * B + beta * C
    // Using cublasSgemm for single-precision matrices
    /* NOTE: since cuBLAS expects column-major order by default,
    I turned the transpose option off with arguments CUBLAS_OP_N*/
    cublasStatus_t status = cublasSgemm(handle, 
                                        CUBLAS_OP_N, CUBLAS_OP_N, 
                                        n, n, n, 
                                        &alpha, 
                                        A, n, 
                                        B, n, 
                                        &beta, 
                                        C, n);


    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cublasSgemm failed!" << std::endl;
    }

    // Synchronize the device
    cudaDeviceSynchronize();
}