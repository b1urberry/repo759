#include "cluster.h"
#include <cmath>
#include <iostream>
#include <omp.h>

// this function does a parallel for loop that calculates the total
// distance between a thread's local center and the data points in its
// corresponding partition.

// it takes a sorted array "arr" of length n, and uses static scheduling
// so that each thread works on its own partition of data.

// t - number of threads.
// "centers" - an array of local center positions; it has length t.
// "dists" - an array that stores the calculated distances; it has length t.
// (if you use padding to resolve the issue then the lengths of these arrays can
// change accordingly)

// Example input: arr = [0, 1, 3, 4, 6, 6, 7, 8], n = 8, t = 2.
// centers = [2, 6] (this is calculated in task1.cpp).
// Expected results: dists = [6, 3].
// 6 = |0-2| + |1-2| + |3-2| + |4-2|; 3 = |6-6| + |6-6| + |7-6| + |8-6|.

const int CACHE_LINE_SIZE = 64;
const int PADDING = CACHE_LINE_SIZE / sizeof(float);

void cluster(const size_t n, const size_t t, const float *arr, const float *centers, float *dists) {
  // Adjust the array size for padding.
  float padded_dists[t * PADDING] = {0};

  #pragma omp parallel num_threads(t)
  {
    unsigned int tid = omp_get_thread_num();
    
    // Use the padded array with an offset to avoid false sharing.
    float* local_dists = padded_dists + tid * PADDING;
    
    #pragma omp for
    for (size_t i = 0; i < n; i++) {
      // Each thread writes to a separate part of the array.
      *local_dists += std::fabs(arr[i] - centers[tid]);
    }
  }

  // After the parallel section, we need to collect the results
  // from the padded array back to the original dists array.
  for (size_t i = 0; i < t; i++) {
    dists[i] = padded_dists[i * PADDING];
  }
}