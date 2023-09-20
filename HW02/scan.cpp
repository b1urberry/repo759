#include "scan.h"
#include <iostream>

// Function definition
void scan(float* arr, const int arr_size, const char operation) {

    if (operation == '+') {
        for (int i = 1; i < arr_size; i++) {
            arr[i] += arr[i-1];
        }
    }
    else if (operation == '*') {
        for (int i = 1; i < arr_size; i++) {
            arr[i] *= arr[i-1];
        }
    }
    
    return;
}
