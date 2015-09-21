// standard libraries
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// OpenCL
#include <CL/cl.h>

// Our stuff
#include "utils.h"
#include "ndarray.h"

void range(double * data, int n) {
    for (int i = 0; i < n ; i++) {
        data[i] = (double) i;
    }
}

ndarray * sample_a() {
    double data[24];
    range(data, 24);
    index_t strides[4] = { 12, 6 ,2 ,1 };
    index_t shape[4] = { 2, 2, 3, 2 };
    return ndarray_constructor(data, 4, shape, strides);
}

ndarray * sample_b() {
    double data[24];
    range(data, 24);
    index_t strides[4] = { 12, 6, 1, 3 };
    index_t shape[4] = { 2, 2, 3, 2 };
    return ndarray_constructor(data, 4, shape, strides);
}

int main() {
    gpu_matrix_init();
    ndarray * x = sample_a();
    ndarray * y = sample_b();
    ndarray * arr = ndarray_add(x, y);
    printf("dimensions: %u\n", arr->ndims);
    for (int i = 0 ; i < 12 ; i++) { 
        printf("%f ", arr->data[i]);
    }
    puts("");
}
