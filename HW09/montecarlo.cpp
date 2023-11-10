#include <cstddef>
#include <omp.h>

// this function returns the number of points that lay inside
// a circle using OpenMP parallel for. 
// You also need to use the simd directive.

// x - an array of random floats in the range [-radius, radius] with length n.
// y - another array of random floats in the range [-radius, radius] with length n.

int montecarlo(const size_t n, const float *x, const float *y, const float radius) {
    int incircle = 0;
    // #pragma omp parallel for reduction(+:incircle)
    #pragma omp simd reduction(+:incircle)
    for (size_t i = 0; i < n; i ++) {
        float a = x[i];
        float b = y[i];
        float d2c_squared = a * a + b * b;
        if (d2c_squared <= radius * radius) {
            incircle ++;
        }
    }
    return incircle;
}