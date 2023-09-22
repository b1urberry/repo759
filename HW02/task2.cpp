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

    // make sure there are 3 command-line arguments
    if (argc != 3) {
        if (argc == 2 && *argv[1] == 't') {
            // if this case, we do testing
            const int n = 4;
            const int m = 3;
            float w[m*m] = {0,0,1,0,1,0,1,0,0};
            float f[n*n] = {1,3,4,8,6,5,2,4,3,4,6,8,1,4,5,2};
            float* g = new float[16];
            convolve(f, g, n, w, m);

            for (int i = 0; i < n; i ++) {
                for (int j = 0; j < n; j ++) {
                    cout << g[i*n+j] << " ";
                }
                cout<< endl;
            }
            delete [] g;
            return 0;
        }
        else {
            return 1;
        }
    }
    
    const int n = atoi(argv[1]);
    const int m = atoi(argv[2]);

    // initialize matrices stored as arrays in row-major order
    float* w = new float[m*m];
    float* f = new float[n*n];
    float* g = new float[n*n];

    // filling the mask(w) and image(f) matrices with random floats    
    std::mt19937 rng(static_cast<unsigned int>(std::time(nullptr))); // seed the random number generator
    std::uniform_real_distribution<float> dist1(-1.0, 1.0); // uniform distribution between -1.0 and 1.0
    std::uniform_real_distribution<float> dist2(-10.0, 10.0); // uniform distribution between -10.0 and 10.0

    for (int i = 0; i < m*m; i++) {
        w[i] = dist1(rng);
        // cout << w[i] << "\n";
    }

    for (int i = 0; i < n*n; i++) {
        f[i] = dist2(rng);
        // cout << f[i] << "\n";
    }

    // intialize variables for timing the function
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;
    

    start = high_resolution_clock::now(); // Get the starting timestamp
    convolve(f, g, n, w, m);
    end = high_resolution_clock::now(); // Get the ending timestamp

    // Convert the calculated duration to a double using the standard library
    duration_sec = std::chrono::duration_cast<duration <double, std::milli> >(end - start);

    cout << duration_sec.count() << "\n";
    cout << g[0] << "\n";
    cout << g[n*n-1] << endl;
    delete [] f;
    delete [] w;
    delete [] g;

    return 0;
}
