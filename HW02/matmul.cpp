#include <iostream>
#include <vector>

#include "matmul.h"

using namespace std;


void mmul1(const double* A, const double* B, double* C, const unsigned int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            for (int k = 0; k < n; k++)
            {
                C[i*n+j] += A[i*n+k] * B[k*n+j];
            }   
        }   
    }
}


void mmul2(const double* A, const double* B, double* C, const unsigned int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            for (int k = 0; k < n; k++)
            {
                C[i*n+k] += A[i*n+j] * B[j*n+k];
            }   
        }   
    }
}


void mmul3(const double* A, const double* B, double* C, const unsigned int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            for (int k = 0; k < n; k++)
            {
                C[j*n+k] += A[j*n+i] * B[i*n+k];
            }   
        }   
    }
}


void mmul4(const std::vector<double>& A, const std::vector<double>& B, double* C, const unsigned int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            for (int k = 0; k < n; k++)
            {
                C[i*n+j] += A[i*n+k] * B[k*n+j];
            }   
        }   
    }
}