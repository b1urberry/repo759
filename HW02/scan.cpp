#include "scan.h"
#include <cstring>
#include <iostream>
#include <cstdlib>

// Function definition
void scan(const float *arr, float *output, std::size_t n) {

    std::memcpy(output, arr, n*sizeof(float)); 

    for (int i = 1; i < n; i++) {
            output[i] += output[i-1];
    }

    return;
}
