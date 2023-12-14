#include "optimize.h"

// optimize1 will be the same as reduce4 function in slide 8.
void optimize1(vec *v, data_t *dest){
    int length = vec_length(v);
    data_t *d = get_vec_start(v);
    data_t temp = IDENT;
    for (int i = 0; i < length; i++)
        temp = temp OP d[i];
    *dest = temp;
}

// optimize2 will be the same as unroll2a reduce function in slide 19.
void optimize2(vec *v, data_t *dest){
    int length = vec_length(v);
    int limit = length - 1;
    data_t *d = get_vec_start(v);
    data_t x = IDENT;
    int i;
    // reduce 2 elements at a time
    for (i = 0; i < limit; i += 2) {
        x = (x OP d[i]) OP d[i + 1];
    }
    // Finish any remaining elements
    for (; i < length; i++) {
        x = x OP d[i];
    }
    *dest = x;
}

// optimize3 will be the same as unroll2aa reduce function in slide 21.
void optimize3(vec *v, data_t *dest){
    int length = vec_length(v);
    int limit = length - 1;
    data_t *d = get_vec_start(v);
    data_t x = IDENT;
    int i;
    /* reduce 2 elements at a time */
    for (i = 0; i < limit; i += 2) {
        x = x OP (d[i] OP d[i + 1]);
    }
    /* Finish any remaining elements */
    for (; i < length; i++) {
        x = x OP d[i];
    }
    *dest = x;
}

// optimize4 will be the same as unroll2a reduce function in slide 24.
void optimize4(vec *v, data_t *dest){
    long length = vec_length(v);
    long limit = length - 1;
    data_t *d = get_vec_start(v);
    data_t x0 = IDENT;
    data_t x1 = IDENT;
    long i;
    // reduce 2 elements at a time
    for (i = 0; i < limit; i += 2) {
        x0 = x0 OP d[i];
        x1 = x1 OP d[i + 1];
    }
    // Finish any remaining elements
    for (; i < length; i++) {
        x0 = x0 OP d[i];
    }
    *dest = x0 OP x1;
}

// optimize5 will be similar to reduce4, but with 
// K = 3 and L = 3, where K and L are the parameters defined in slide 27.
void optimize5(vec *v, data_t *dest){
    int length = vec_length(v);
    data_t *d = get_vec_start(v);
    data_t temp = IDENT;
    for (int i = 0; i < length; i++)
        temp = temp OP d[i];
    *dest = temp;
}