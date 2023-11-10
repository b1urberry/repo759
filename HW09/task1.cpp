#include <iostream>
#include <algorithm>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <omp.h>

// Assume cluster.h defines the `cluster` function as provided earlier.
#include "cluster.h" 

using std::chrono::high_resolution_clock;
using std::chrono::duration;

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " n t" << std::endl;
        return 1;
    }

    size_t n = std::atoi(argv[1]);
    size_t t = std::atoi(argv[2]);
    std::vector<float> arr(n);
    std::vector<float> centers(t);
    std::vector<float> dists(t, 0.0f);

    // Initialize random number generator
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    // Fill array with random floats
    for (size_t i = 0; i < n; ++i) {
        arr[i] = static_cast<float>(std::rand()) / (static_cast<float>(RAND_MAX / n));
    }

    // Sort the array
    std::sort(arr.begin(), arr.end());

    // Fill centers array
    for (size_t i = 0; i < t; ++i) {
        centers[i] = (static_cast<float>((2 * i) + 1) / static_cast<float>(2 * t)) * n;
    }

    // intialize variables for timing the function
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;
    // Measure time taken by cluster function
    start = high_resolution_clock::now(); // Get the starting timestamp
    cluster(n, t, arr.data(), centers.data(), dists.data());
    end = high_resolution_clock::now(); // Get the ending timestamp

    // Calculate maximum distance
    float max_distance = *std::max_element(dists.begin(), dists.end());
    size_t max_distance_id = static_cast<size_t>(std::distance(dists.begin(), std::max_element(dists.begin(), dists.end())));

    // Print maximum distance
    std::cout << max_distance << std::endl;
    // Print partition ID
    std::cout << max_distance_id << std::endl;
    // Print time taken in milliseconds
    duration_sec = std::chrono::duration_cast<duration <double, std::milli> >(end - start);
    std::cout << duration_sec.count() << std::endl;

    return 0;
}
