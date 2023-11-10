#include "montecarlo.h"
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

    size_t n = std::atoi(argv[1]); 
    size_t t = std::atoi(argv[2]); // number of threads
    omp_set_num_threads(t);

    float r  = 1.0;
    float* x = new float[n];
    float* y = new float[n];

    // filling the mask(w) and image(f) matrices with random floats    
    std::mt19937 rng(static_cast<unsigned int>(std::time(nullptr))); // seed the random number generator
    std::uniform_real_distribution<float> dist1(-r, r); // uniform distribution between -1.0 and 1.0
    std::uniform_real_distribution<float> dist2(-r, r); // uniform distribution between -10.0 and 10.0

    for (size_t i = 0; i < n; i++) {
        x[i] = dist1(rng);
        // cout << x[i] << "\n";
    }

    for (size_t i = 0; i < n; i++) {
        y[i] = dist2(rng);
        // cout << y[i] << "\n";
    }

    // intialize variables for timing the function
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;
    
    start = high_resolution_clock::now(); // Get the starting timestamp
    int result = montecarlo(n, x, y, r);
    end = high_resolution_clock::now(); // Get the ending timestamp

    // Convert the calculated duration to a double using the standard library
    duration_sec = std::chrono::duration_cast<duration <double, std::milli> >(end - start);

    // estimate pi
    float pi = 4.0 * (float)result / (float)n;

    cout << pi << "\n";
    cout << duration_sec.count() << endl;
    delete [] x;
    delete [] y;

    return 0;
}
