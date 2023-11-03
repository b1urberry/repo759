#include <iostream>
#include <cstdlib>
#include <ctime>
#include <ostream>
#include <random>
#include <chrono>
#include <cmath>
#include <vector>

#include "matmul.h"

// namespace shortcuts
using namespace std;
using std::cout;
using std::endl;
using std::chrono::high_resolution_clock;
using std::chrono::duration;
using std::vector;


int main(int argc, char* argv[]) {

    if (argc != 3) {
        return 1;
    }

    size_t n = std::atoi(argv[1]); // dimension of the matrices are n*n
    size_t t = std::atoi(argv[2]); // number of threads
    omp_set_num_threads(t);

    // declare matrix variables and allocate memory
    float *A = (float *)malloc(sizeof(float)*n*n);
    float *B = (float *)malloc(sizeof(float)*n*n);
    float *C1 = (float *)malloc(sizeof(float)*n*n);

    // filling the mask(w) and image(f) matrices with random floats    
    std::mt19937 rng(static_cast<unsigned int>(std::time(nullptr))); // seed the random number generator
    std::uniform_real_distribution<float> dist1(-1.0, 1.0); // uniform distribution between -1.0 and 1.0
    std::uniform_real_distribution<float> dist2(-1.0, 1.0); // uniform distribution between -1.0 and 1.0

    // fill up A with random numbers between -1 and 1
    for (size_t i = 0; i < n*n; i++)
    {
        float r1 = dist1(rng);
        float r2 = dist2(rng);
        A[i] = r1;
        B[i] = r2;
        C1[i] = 0.0;
    }

    // intialize variables for timing the function
    high_resolution_clock::time_point start1;
    high_resolution_clock::time_point end1;
    duration<float, std::milli> duration_sec1;

    start1 = high_resolution_clock::now(); 
    mmul(A, B, C1, n);
    end1 = high_resolution_clock::now(); 
    duration_sec1 = std::chrono::duration_cast<duration <float, std::milli> >(end1 - start1);

    cout << C1[0] << "\n";
    cout << C1[n*n-1] << "\n";
    cout << duration_sec1.count() << "\n";

    free(A);
    free(B);
    free(C1);

    return 0;
}