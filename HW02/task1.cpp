#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <cmath>

#include "scan.h"

// namespace shortcuts
using namespace std;
using std::cout;
using std::endl;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

int main(int argc, char* argv[]) {

    if (argc != 2) {
        return 1;
    }

    //
    //
    // i) Creates an array of n random float numbers between -1.0 and 1.0, n read as the first command line argument.
    int n = std::atoi(argv[1]);
    float* arr = (float*)malloc(n * sizeof(float));
    float* result = (float*)malloc(n * sizeof(float));
    srand(static_cast<unsigned int>(time(nullptr))); // random seed based on current system time

    // populate the array of n floats
    for (int i = 0; i < n; i++) {
        arr[i] = -1.0f + (float)rand() /( (float)RAND_MAX/2.0f);
    }

    //
    // 
    // ii) Scans the array using your scan function.
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;
    
    start = high_resolution_clock::now(); // Get the starting timestamp
    scan(arr, result, n);
    end = high_resolution_clock::now(); // Get the ending timestamp

    // Convert the calculated duration to a double using the standard library
    duration_sec = std::chrono::duration_cast<duration <double, std::milli> >(end - start);
    
    // // quick test to see if scan works
    // for (int i = 0; i < n; i ++) {
    //     cout << arr[i] << "\n";
    // }
    // cout << "========\n";
    // for (int i = 0; i < n; i ++) {
    //     cout << result[i] << "\n";
    // }

    // Durations are converted to milliseconds already thanks to std::chrono::duration_cast
    cout << duration_sec.count() << "\n";
    cout << result[0] << "\n";
    cout << result[n-1] << endl;
    free(arr);
    free(result);
    return 0;
}