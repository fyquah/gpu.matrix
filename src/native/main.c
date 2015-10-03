// standard libraries
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

// Our stuff
#include "utils.h"
#include "ndarray.h"
#include "vector.h"
#define LENGTH 400000000

double data_a[LENGTH], data_b[LENGTH];

int main() {
    vector a, b;
    gpu_matrix_init();
    a.length = LENGTH;
    b.length = LENGTH;
    a.stride = 1;
    b.stride = 1;
    a.data = data_a;
    b.data = data_b;

    for (int i = 0 ; i < LENGTH ; i++) {
        a.data[i] = 1.20 * i;
        b.data[i] = 35.0 * i;
    }

    struct timeval stop, start;
    gettimeofday(&start, NULL);
    vector * c = gpu_matrix_vector_axpy(&a, 12.0, &b);
    gettimeofday(&stop, NULL);
    printf("Took %ld microseconds\n", stop.tv_usec + stop.tv_sec * 1000000 - 
            (start.tv_usec + start.tv_sec * 1000000));
    gettimeofday(&start, NULL);
    
    printf("%ld\n", c->length);
    for (int i = 0 ; i < LENGTH ; i++) {
        if(c->data[i] != a.data[i] * 12.0 + b.data[i]) {
            puts("WRONG");
            exit(1);
        }
    }

    gettimeofday(&stop, NULL);
    printf("%ld microseconds\n", stop.tv_usec + stop.tv_sec * 1000000 - 
            (start.tv_usec + start.tv_sec * 1000000));
    puts("RIGHT!");


    return 0;
}
