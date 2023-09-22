#include <iostream>

#include "convolution.h"


int xy2index (const int x, const int y, const int n);

void convolve(const float *image, float *output, std::size_t n, const float *mask, std::size_t m){

    const int k = (m - 1) / 2;
    
    for (int x = 0; x < n; x ++) {
        for (int y = 0; y < n; y ++) {

            float sum = 0;

            for (int i = 0; i < m; i ++) {
                for (int j = 0; j < m; j ++) {
                    // float* w_ij_pointer = w + i * m + j;
                    float w_ij = mask[xy2index(i, j, m)];
                    
                    int check_num = 0;
                    const int new_x = x+i-k;
                    const int new_y = y+j-k;

                    check_num += new_x < 0 or new_x >= n ? 1 : 0;
                    check_num += new_y < 0 or new_y >= n ? 1 : 0;  

                    float f_result = 
                        check_num == 0 ? 
                        image[xy2index(x+i-k, y+j-k, n)] :
                        (check_num == 1 ? 1 : 0);                   
                    sum += w_ij * f_result;
                }
            }

            output[xy2index(x, y, n)] = sum;
            
        }
    }

    return;

}

int xy2index (const int x, const int y, const int size) {
    return x * size + y;
}
