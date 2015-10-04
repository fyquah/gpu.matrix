// standard libraries
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

// Our stuff
#include "utils.h"
#include "ndarray.h"
#include "vector.h"
#define LENGTH 4194304 
#define ENABLE_PROFILING 1


int main() {
    vector a, b, *c;
    int d = 123;
    double *data_a, *data_b, output;

    gpu_matrix_init();

    data_a = malloc(sizeof(double) * LENGTH);
    data_b = malloc(sizeof(double) * LENGTH);
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

    output = gpu_matrix_vector_asum(&a);

    gettimeofday(&start, NULL);
    // CODE HERE
    output = 0.0;
    for (int i = 0 ; i < LENGTH ; i++) {
        output = output + a.data[i];
    }
    // END OF CODE

    gettimeofday(&stop, NULL);
    float time_taken = (stop.tv_usec + stop.tv_sec * 1000000 - 
            (start.tv_usec + start.tv_sec * 1000000)) / 1000.0;
    printf("Pure SW took %f milliseconds\n", time_taken);
    

    return 0;
}
