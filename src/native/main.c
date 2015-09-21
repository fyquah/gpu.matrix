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

int main() {
    gpu_matrix_init();
    ndarray * arr = malloc(sizeof(ndarray));
    double data[6] = { 0, 2.22, 3.0, 1.0, 2.0, 8.0 };
    double data_b[6] = { 0, 2.22, 3.0, 1.0, 2.0, 8.0 };
    index_t strides[2] = { 3, 1 };
    index_t shape[2] = { 2, 3 };
    index_t a [3] = { 1 ,2,3};
    index_t b[3] = { 1,2,3};
    arr->data = data;
    arr->ndims = 2;
    arr->strides = strides;
    arr->shape = shape;

    printf("-> %d\n", array_index_t_is_equal(a, b, 3));

    ndarray_add_scalar_bang(arr, 123.0);
    printf("dimensions: %lu\n", arr->ndims);
    for (int i = 0 ; i < 6 ; i++) { 
        printf("%f ", arr->data[i]);
    }
    puts("");
}
