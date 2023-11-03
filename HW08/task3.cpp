#include <iostream>
#include <cstdlib>
#include <ctime>
#include <ostream>
#include <random>
#include <chrono>
#include <cmath>

#include "msort.h"

// namespace shortcuts
using namespace std;
using std::cout;
using std::endl;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

int main(int argc, char* argv[]){

    if (argc != 4) {
        return 1;
    }

    size_t n = std::atoi(argv[1]); // dimension of the matrices are n*n
    size_t t = std::atoi(argv[2]); // number of threads
    size_t ts = std::atoi(argv[3]); // threashold under which we just use serial sort
    omp_set_num_threads(t);

    int* arr = new int[n];

    // fill arr with random into between -1000 and 1000
    std::mt19937 rng(static_cast<unsigned int>(std::time(nullptr))); // seed the random number generator
    std::uniform_int_distribution<> dist(-1000, 1000); 
    for (size_t i = 0; i < n; i++) {
        arr[i] = dist(rng);
    }

    // // for testing
    // cout << "unsorted" << "\n";
    // for (size_t i = 0; i < n; i++) {
    //     cout << arr[i] << ",";
    // }
    // printf("\n");

    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;
    
    start = high_resolution_clock::now(); // Get the starting timestamp
    msort(arr, n, ts);
    end = high_resolution_clock::now(); // Get the ending timestamp

    // Convert the calculated duration to a double using the standard library
    duration_sec = std::chrono::duration_cast<duration <double, std::milli> >(end - start);

    // // for testing
    // cout << "sorted" << "\n";
    // for (size_t i = 0; i < n; i++) {
    //     cout << arr[i] << ",";
    // }
    // printf("\n");

    cout << arr[0] << "\n";
    cout << arr[n-1] << endl;
    cout << duration_sec.count() << "\n";

    delete[] arr;

    return 0;
}