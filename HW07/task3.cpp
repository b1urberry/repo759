// task3.cpp

#include <iostream>
#include <omp.h>

long long factorial(int n) {
    long long result = 1;
    for (int i = 1; i <= n; ++i) {
        result *= i;
    }
    return result;
}

int main() {
    // Setting the number of threads to 4
    omp_set_num_threads(4);

    long long factorials[8];

    #pragma omp parallel
    {
        // Print the number of threads only once by the master thread
        #pragma omp master
        {
            std::cout << "Number of threads: " << omp_get_num_threads() << std::endl;
        }

        // Each thread introduces itself using a critical section
        #pragma omp critical
        {
            std::cout << "I am thread No. " << omp_get_thread_num() << std::endl;
        }

        // Compute factorials in parallel
        #pragma omp for
        for (int i = 1; i <= 8; ++i) {
            factorials[i-1] = factorial(i);
        }
    }

    // Print the factorials outside the parallel section to avoid interleaved outputs
    for (int i = 1; i <= 8; ++i) {
        std::cout << i << "!=" << factorials[i-1] << std::endl;
    }

    return 0;
}
