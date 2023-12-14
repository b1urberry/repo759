#include <iostream>
#include <cstdlib>
#include <ctime>
#include <ostream>
#include <random>
#include <chrono>
#include <cmath>
#include <vector>

#include "optimize.h"

// namespace shortcuts
using namespace std;
using std::cout;
using std::endl;
using std::chrono::high_resolution_clock;
using std::chrono::duration;
using std::vector;


int main(int argc, char* argv[]) {

    if (argc != 2) {
        return 1;
    }

    // creat an array arr of length n, fill it with random ints between -10 and 10
    size_t n = atoi(argv[1]); 
    vector<data_t> arr(n);
    std::mt19937 rng(static_cast<unsigned int>(std::time(nullptr))); 
    std::uniform_int_distribution<int> dist(-10,10); 
    for (size_t i = 0; i < n; i++)
    {
        arr[i] = dist(rng);
    }

    vec v = {n, arr};  // initialize a vec v with length n and data arr
    data_t dest;// initialize the result pointer, dest
    
    // intialize variables for timing the function
    high_resolution_clock::time_point start1;
    high_resolution_clock::time_point end1;
    high_resolution_clock::time_point start2;
    high_resolution_clock::time_point end2;
    high_resolution_clock::time_point start3;
    high_resolution_clock::time_point end3;
    high_resolution_clock::time_point start4;
    high_resolution_clock::time_point end4;
    high_resolution_clock::time_point start5;
    high_resolution_clock::time_point end5;
    duration<float, std::milli> duration_sec1;
    duration<float, std::milli> duration_sec2;
    duration<float, std::milli> duration_sec3;
    duration<float, std::milli> duration_sec4;
    duration<float, std::milli> duration_sec5;

    start1 = high_resolution_clock::now(); 
    optimize1(&v, &dest);
    end1 = high_resolution_clock::now(); 
    duration_sec = std::chrono::duration_cast<duration <float, std::milli> >(end1 - start1);
    cout << dest << "\n";
    cout << duration_sec.count() << "\n";
    start2 = high_resolution_clock::now(); 
    optimize1(&v, &dest);
    end2 = high_resolution_clock::now(); 
    duration_sec = std::chrono::duration_cast<duration <float, std::milli> >(end2 - start2);
    cout << dest << "\n";
    cout << duration_sec.count() << "\n";
    start3 = high_resolution_clock::now(); 
    optimize1(&v, &dest);
    end3 = high_resolution_clock::now(); 
    duration_sec = std::chrono::duration_cast<duration <float, std::milli> >(end3 - start1);
    cout << dest << "\n";
    cout << duration_sec.count() << "\n";
    start4 = high_resolution_clock::now(); 
    optimize1(&v, &dest);
    end4 = high_resolution_clock::now(); 
    duration_sec = std::chrono::duration_cast<duration <float, std::milli> >(end4 - start1);
    cout << dest << "\n";
    cout << duration_sec.count() << "\n";
    start5 = high_resolution_clock::now(); 
    optimize1(&v, &dest);
    end5 = high_resolution_clock::now(); 
    duration_sec = std::chrono::duration_cast<duration <float, std::milli> >(end5 - start1);
    cout << dest << "\n";
    cout << duration_sec.count() << "\n";

    return 0;
}