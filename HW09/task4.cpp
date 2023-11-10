#include "convolve.h"
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <ostream>
#include <random>
#include <chrono>
#include <cmath>

using namespace std;
using std::cout;
using std::endl;
using std::chrono::high_resolution_clock;
using std::chrono::duration;


int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <n>" << std::endl;
        return 1;
    }

    // Parse the command line argument
    int n = std::atoi(argv[1]);
    int m = 3;

    std::vector<float> image(n * n);
    std::vector<float> mask(m * m);
    std::vector<float> output(n * n);

    // filling the mask(w) and image(f) matrices with random floats    
    std::mt19937 rng(static_cast<unsigned int>(std::time(nullptr))); // seed the random number generator
    std::uniform_real_distribution<float> dist1(-1.0, 1.0); // uniform distribution between -1.0 and 1.0
    std::uniform_real_distribution<float> dist2(-10.0, 10.0); // uniform distribution between -10.0 and 10.0

    for (int i = 0; i < m*m; i++) {
        mask[i] = dist1(rng);
        // cout << w[i] << "\n";
    }

    for (int i = 0; i < n*n; i++) {
        image[i] = dist2(rng);
        // cout << f[i] << "\n";
    }


    // intialize variables for timing the function
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;
    
    start = high_resolution_clock::now();
    convolve(image.data(), output.data(), n, mask.data(), m);
    end = high_resolution_clock::now();
    
    duration_sec = std::chrono::duration_cast<duration <double, std::milli> >(end - start);

    // Print the time taken in milliseconds
    std::cout << duration_sec.count() << std::endl;

    return 0;
}
