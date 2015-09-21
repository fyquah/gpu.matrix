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
    index_t strides[2] = { 3, 1 };
    index_t shape[2] = { 2, 3 };
    arr->data = data;
    arr->ndims = 2;
    arr->strides = strides;
    arr->shape = shape;
    
    arr = ndarray_add(arr, arr);
    printf("dimensions: %lu\n", arr->ndims);
    for (int i = 0 ; i < 6 ; i++) { 
        printf("%f ", arr->data[i]);
    }
    puts("");
}
