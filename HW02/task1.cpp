#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <cmath>

// namespace shortcuts
using std::cout;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

int main(int argc, char* argv[]) {

    if (argc != 2) {
        return 1;
    }

    //
    //
    // i) Creates an array of n random float numbers between -1.0 and 1.0, n read as the first command line argument.
    int n = atoi(argv[1]);
    float arr[n]; 
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
    float result_arr[n] = scan(arr);
    end = high_resolution_clock::now(); // Get the ending timestamp

    // Convert the calculated duration to a double using the standard library
    duration_sec = std::chrono::duration_cast<duration <double, std::milli> >(end - start);
    
    // Durations are converted to milliseconds already thanks to std::chrono::duration_cast
    cout << "Total time: " << duration_sec.count() << "ms\n";


    return 0;
}