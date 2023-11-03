#include "convolution.h"

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <ostream>
#include <random>
#include <chrono>
#include <cmath>

// namespace shortcuts
using namespace std;
using std::cout;
using std::endl;
using std::chrono::high_resolution_clock;
using std::chrono::duration;


int main(int argc, char* argv[]){

    if (argc != 3) {
        return 1;
    }

    size_t n = std::atoi(argv[1]); // dimension of the matrices are n*n
    size_t t = std::atoi(argv[2]); // number of threads
    omp_set_num_threads(t);

    // initialize matrices stored as arrays in row-major order
    float* w = new float[3*3];
    float* f = new float[n*n];
    float* g = new float[n*n];

    // filling the mask(w) and image(f) matrices with random floats    
    std::mt19937 rng(static_cast<unsigned int>(std::time(nullptr))); // seed the random number generator
    std::uniform_real_distribution<float> dist1(-1.0, 1.0); // uniform distribution between -1.0 and 1.0
    std::uniform_real_distribution<float> dist2(-10.0, 10.0); // uniform distribution between -10.0 and 10.0

    for (size_t i = 0; i < 3*3; i++) {
        w[i] = dist1(rng);
        // cout << w[i] << "\n";
    }

    for (size_t i = 0; i < n*n; i++) {
        f[i] = dist2(rng);
        // cout << f[i] << "\n";
    }

    // intialize variables for timing the function
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;
    

    start = high_resolution_clock::now(); // Get the starting timestamp
    convolve(f, g, n, w, 3);
    end = high_resolution_clock::now(); // Get the ending timestamp

    // Convert the calculated duration to a double using the standard library
    duration_sec = std::chrono::duration_cast<duration <double, std::milli> >(end - start);

    cout << g[0] << "\n";
    cout << g[n*n-1] << endl;
    cout << duration_sec.count() << "\n";
    delete [] f;
    delete [] w;
    delete [] g;

    return 0;
}
