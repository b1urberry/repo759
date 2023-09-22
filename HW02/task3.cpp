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

    int n = 1024;

    // declare matrix variables and allocate memory
    double *A = (double *)malloc(sizeof(double)*n*n);
    double *B = (double *)malloc(sizeof(double)*n*n);
    double *C1 = (double *)malloc(sizeof(double)*n*n);
    double *C2 = (double *)malloc(sizeof(double)*n*n);
    double *C3 = (double *)malloc(sizeof(double)*n*n);
    double *C4 = (double *)malloc(sizeof(double)*n*n);
    vector<double> A_vec;
    vector<double> B_vec;

    // filling the mask(w) and image(f) matrices with random floats    
    std::mt19937 rng(static_cast<unsigned int>(std::time(nullptr))); // seed the random number generator
    std::uniform_real_distribution<double> dist1(-1.0, 1.0); // uniform distribution between -1.0 and 1.0
    std::uniform_real_distribution<double> dist2(-1.0, 1.0); // uniform distribution between -1.0 and 1.0

    // fill up A with random numbers between -1 and 1
    for (size_t i = 0; i < n*n; i++)
    {
        double r1 = dist1(rng);
        double r2 = dist2(rng);
        A[i] = r1;
        B[i] = r2;
        A_vec.push_back(r1);
        B_vec.push_back(r2);
        C1[i] = 0.0;
        C2[i] = 0.0;
        C3[i] = 0.0;
        C4[i] = 0.0;
    }



    // intialize variables for timing the function
    high_resolution_clock::time_point start1;
    high_resolution_clock::time_point end1;
    duration<double, std::milli> duration_sec1;
    high_resolution_clock::time_point start2;
    high_resolution_clock::time_point end2;
    duration<double, std::milli> duration_sec2;
    high_resolution_clock::time_point start3;
    high_resolution_clock::time_point end3;
    duration<double, std::milli> duration_sec3;
    high_resolution_clock::time_point start4;
    high_resolution_clock::time_point end4;
    duration<double, std::milli> duration_sec4;

    start1 = high_resolution_clock::now(); 
    mmul1(A, B, C1, n);
    end1 = high_resolution_clock::now(); 
    duration_sec1 = std::chrono::duration_cast<duration <double, std::milli> >(end1 - start1);


    start2 = high_resolution_clock::now(); 
    mmul2(A, B, C2, n);
    end2 = high_resolution_clock::now(); 
    duration_sec2 = std::chrono::duration_cast<duration <double, std::milli> >(end2 - start2);

    start3 = high_resolution_clock::now(); 
    mmul3(A, B, C3, n);
    end3 = high_resolution_clock::now(); 
    duration_sec3 = std::chrono::duration_cast<duration <double, std::milli> >(end3 - start3);

    start4 = high_resolution_clock::now(); 
    mmul4(A_vec, B_vec, C4, n);
    end4 = high_resolution_clock::now(); 

    cout << n << "\n";
    cout << duration_sec1.count() << "\n";
    cout << C1[n*n-1] << "\n";
    cout << duration_sec2.count() << "\n";
    cout << C2[n*n-1] << "\n";
    cout << duration_sec3.count() << "\n";
    cout << C3[n*n-1] << "\n";
    cout << duration_sec4.count() << "\n";
    cout << C4[n*n-1] << "\n";

    delete [] A;
    delete [] B;
    delete [] C1;
    delete [] C2;
    delete [] C3;
    delete [] C4;

    return 0;
}