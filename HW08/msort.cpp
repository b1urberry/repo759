#include <omp.h>
#include <algorithm>  // for std::sort

void merge(int* arr, std::size_t left, std::size_t mid, std::size_t right) {
    std::size_t n1 = mid - left;
    std::size_t n2 = right - mid;

    int* leftArr = new int[n1];
    int* rightArr = new int[n2];

    for(std::size_t i = 0; i < n1; i++)
        leftArr[i] = arr[left + i];
    for(std::size_t j = 0; j < n2; j++)
        rightArr[j] = arr[mid + j];

    std::size_t i = 0, j = 0, k = left;
    while(i < n1 && j < n2) {
        if(leftArr[i] <= rightArr[j]) {
            arr[k] = leftArr[i];
            i++;
        } else {
            arr[k] = rightArr[j];
            j++;
        }
        k++;
    }

    while(i < n1) {
        arr[k] = leftArr[i];
        i++;
        k++;
    }

    while(j < n2) {
        arr[k] = rightArr[j];
        j++;
        k++;
    }

    delete[] leftArr;
    delete[] rightArr;
}

void merge_sort(int* arr, std::size_t left, std::size_t right, const std::size_t threshold) {
    if (right - left <= threshold) {
        std::sort(arr + left, arr + right);  // Use serial sort for small subarrays
        return;
    }

    std::size_t mid = left + (right - left) / 2;

    #pragma omp task  // Create a parallel task for the left half
    merge_sort(arr, left, mid, threshold);

    #pragma omp task  // Create a parallel task for the right half
    merge_sort(arr, mid, right, threshold);

    #pragma omp taskwait  // Wait for both tasks to complete
    merge(arr, left, mid, right);  // Merge the two sorted halves
}

void msort(int* arr, const std::size_t n, const std::size_t threshold) {
    #pragma omp parallel  // Start parallel region
    {
        #pragma omp single  // Ensure merge_sort is called only once
        {
            merge_sort(arr, 0, n, threshold);
        }
    }
}
