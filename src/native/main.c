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
    for (int i = 0 ; i < 24; i++) {
        data[i] *= 2.0;
    }
    index_t strides[2] = { 4, 1 };
    index_t shape[2] = { 6, 4 };
    return ndarray_constructor(data, 2, shape, strides);
}

ndarray * sample_c() {
    double data[12];
    range(data, 12);
    for (int i = 0 ; i < 12; i++) {
        data[i] *= 2;
    }
    index_t strides[2] = { 3, 1 };
    index_t shape[2] = { 4, 3 };
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
    ndarray * mat_a = sample_a();
    ndarray * mat_c = sample_c();
    ndarray * output = ndarray_mmul(mat_a, mat_c);

    puts("[");
    for (int i = 0 ; i < mat_a->shape[0] ; i++) {
        for (int j = 0 ; j < mat_a->shape[1] ; j++) {
            printf("%.2f ", mat_a->data[i * mat_a->shape[1] + j]);
        }
        puts("");
    }
    puts("]");

    puts("[");
    for (int i = 0 ; i < mat_c->shape[0] ; i++) {
        for (int j = 0 ; j < mat_c->shape[1] ; j++) {
            printf("%.2f ", mat_c->data[i * mat_c->shape[1] + j]);
        }
        puts("");
    }
    puts("]");

    puts("[");
    for (int i = 0 ; i < output->shape[0] ; i++) {
        for (int j = 0 ; j < output->shape[1] ; j++) {
            printf("%.2f ", output->data[i * output->shape[1] + j]);
        }
        puts("");
    }
    puts("]");
    puts("");

    return 0;
}
