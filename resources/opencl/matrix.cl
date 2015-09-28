#include <types.h>

__kernel void mmul (
    __global double * data_x,
    __global index_t * shape_x,
    __global index_t * strides_x,
    __global double * data_y,
    __global index_t * shape_y,
    __global index_t * strides_y,
    __global double * data_output,
    __global index_t * strides_output
) {
    // note : width refers to number of columns 
    index_t row = get_global_id (0);
    index_t col = get_global_id (1);

    int sum = 0.0;

    for (index_t i = 0 ; i < shape_x[1]; i++) {
        sum += data_x[row * strides_x[0] + i * strides_x[1]] * 
                data_y[i * strides_y[0] + col * strides_y[1]];  
    }

    data_output[row * strides_output[0] + col * strides_output[1]] = sum;
}
