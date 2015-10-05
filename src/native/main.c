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
        a.data[i] = i * 1.2;
        b.data[i] = i * 1.3;
    }

    output = gpu_matrix_vector_asum(&b);
    output = gpu_matrix_vector_asum(&b);
    printf("%.2f", output);

    return 0;
}
