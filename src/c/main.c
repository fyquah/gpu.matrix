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
    ndarray * arr = malloc(sizeof(ndarray));
    double data[3] = { 4.0, 2.22, 3.0 };
    size_t strides[1] = { 1 };
    size_t shape[1] = { 3 };
    arr->data = data;
    arr->ndims = 1;
    arr->strides = strides;
    arr->shape = shape;
    
    init();  
    ndarray_add_scalar_bang(arr, 1.0);
    printf("dimensions: %d\n", arr->ndims);
    for (int i = 0 ; i < 3 ; i++) {
        printf("%f ", arr->data[i]);
    }
    puts("");
}