// count.cuh

#ifndef COUNT_CUH
#define COUNT_CUH

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/distance.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/reduce.h>

#include "count.cuh"

// Find the unique integers in the array d_in,
// store these integers in values array in ascending order,
// store the occurrences of these integers in counts array.
// values and counts should have equal length.
// Example:
// d_in = [3, 5, 1, 2, 3, 1]
// Expected output:
// values = [1, 2, 3, 5]
// counts = [2, 1, 2, 1]
void count(const thrust::device_vector<int>& d_in,
           thrust::device_vector<int>& values,
           thrust::device_vector<int>& counts) {

    // Copy input to temp values (since thrust::unique_by_key modifies input)
    thrust::device_vector<int> temp_values = d_in;

    // Sort the temp values
    thrust::sort(temp_values.begin(), temp_values.end());

    // Resize output vectors (in the worst case they will be same length as d_in)
    values.resize(d_in.size());
    counts.resize(d_in.size());

    // Use thrust::unique_by_key to get the unique values and their counts
    auto new_end = thrust::reduce_by_key(temp_values.begin(), temp_values.end(),
                                         thrust::constant_iterator<int>(1),
                                         values.begin(),
                                         counts.begin());

    // Resize values and counts vectors to actual size
    values.resize(thrust::distance(values.begin(), new_end.first));
    counts.resize(thrust::distance(counts.begin(), new_end.second));
}

#endif
