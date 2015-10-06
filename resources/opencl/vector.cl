#ifndef VECTOR_CL
#define VECTOR_CL

#include <types.h>

__kernel void vector_mul (
    // The acutla parameters to the kernel
    __global const double * data_x,
    const index_t length_x,
    const index_t stride_x,
    __global const double * data_y,
    const index_t length_y,
    const index_t stride_y,
    // Output memory locations
    __global double * data_output
) {
    index_t global_id = get_global_id (0);
    data_output[global_id] = 
        data_x[global_id * stride_x] *
        data_y[global_id * stride_y];
}

__kernel void vector_mul_bang (
    // The acutla parameters to the kernel
    __global double * data_x,
    const index_t length_x,
    const index_t stride_x,
    __global const double * data_y,
    const index_t length_y,
    const index_t stride_y
) {
    index_t global_id = get_global_id (0);
    data_x[global_id * stride_x] = 
        data_x[global_id * stride_x] *
        data_y[global_id * stride_y];
}

__kernel void vector_square_bang (
    // The acutla parameters to the kernel
    __global double * data_x,
    const index_t length_x,
    const index_t stride_x
) {
    index_t global_id = get_global_id (0);
    double v = data_x[global_id * stride_x];
    data_x[global_id * stride_x] = v * v;
}

__kernel void vector_asum (
    __global double * data,
    const index_t len,
    const index_t stride
) {
    index_t global_id = get_global_id (0);
    index_t first_id = global_id * stride;
    index_t second_id = (global_id + ((len+1) / 2) ) * stride;

    data[first_id] += data[second_id];
}

__kernel void vector_axpy_bang (
    // The actual parameters to the kernel
    __global double * data_x,
    const index_t length_x,
    const index_t stride_x,
    const double alpha,
    __global const double * data_y,
    const index_t length_y,
    const index_t stride_y
) {
    index_t global_id = get_global_id (0);

    data_x[global_id * stride_x] = alpha * 
        data_x[global_id * stride_x] +
        data_y[global_id * stride_y];
}

__kernel void vector_axpy (
    // The actual parameters to the kernel
    __global const double * data_x,
    const index_t length_x,
    const index_t stride_x,
    const double alpha,
    __global const double * data_y,
    const index_t length_y,
    const index_t stride_y,
    // Output memory locations
    __global double * data_output
) {
    index_t global_id = get_global_id (0);

    data_output[global_id] = alpha * 
        data_x[global_id * stride_x] +
        data_y[global_id * stride_y];
}

#endif

