// standard libraries
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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
    index_t strides[2] = { 3, 1 };
    index_t shape[2] = { 6, 3 };
    return ndarray_constructor(data, 2, shape, strides);
}

ndarray * sample_b() {
    double data[4];
    range(data, 4);
    index_t strides[1] = { 1 };
    index_t shape[1] = { 3 };
    return ndarray_constructor(data, 1, shape, strides);
}

int main() {
    gpu_matrix_init();
    ndarray * mat = sample_a();
    ndarray * arr = ndarray_broadcast(sample_b(), mat->ndims, mat->shape);
    printf("dimensions: %u\n", arr->ndims);
    printf("elements count: %u\n", ndarray_elements_count(arr));
    puts("data : ");
    arr = ndarray_add(arr, arr);
    for (int i = 0 ; i < ndarray_elements_count(arr) ; i++) { 
        printf("%f ", arr->data[i]);
    }
    puts("");
    puts("strides: ");
    for (int i = 0 ; i < arr->ndims ; i++) {
        printf("%u ", arr->strides[i]);
    }
    puts("");
}
