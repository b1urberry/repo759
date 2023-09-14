#include <cstdio>
#include <iostream>
#include <sys/types.h>
#include <cstdlib>


int main(int argc, char* argv[]) {

    int n = atoi(argv[1]);
    
    if (argc > 1) {
        for (int i = 0; i <= n; i++) {
            printf("%d ", i);
        }
        std::cout << std::endl;

        for (int i = 0; i <= n; i++) {
            std::cout << i << " " << std::flush;
        }
        std::cout << std::endl;  
    }

    return 0;
}