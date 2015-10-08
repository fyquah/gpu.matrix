#ifndef VECTOR_CL
#define VECTOR_CL

#include <types.h>

__kernel void vector_rot_bang (
    __global double * data_x,
    const index_t length_x,
    const index_t stride_x,
    __global double * data_y,
    const index_t length_y,
    const index_t stride_y,
    const double c,
    const double s
) {
    index_t global_id = get_global_id (0);
    index_t idx_x = global_id * stride_x;
    index_t idx_y = global_id * stride_y;
    double x = data_x[idx_x];
    double y = data_y[idx_y];

    data_x[idx_x] = c * x + s * y;
    data_y[idx_y] = c * y - s * y;
}

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

    data[first_id] = fabs (data[first_id]) + fabs(data[second_id]);
}

__kernel void vector_axpy_bang (
    // The actual parameters to the kernel
    __global double * data_x,
    const index_t length_x,
    const index_t stride_x,
    __global const double * data_y,
    const index_t length_y,
    const index_t stride_y,
    const double alpha
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

__kernel void vector_abs_bang (
    // The actual parameters to the kernel
    __global double * data_x,
    const index_t length_x,
    const index_t stride_x
) {
    index_t global_id = get_global_id (0);
    data_x[global_id * stride_x] = 
        fabs (data_x[global_id * stride_x]);
}

__kernel void vector_max_bang (
    // The actual parameters to the kernel
    __global double * data,
    const index_t len,
    const index_t stride
) {
    index_t global_id = get_global_id (0);
    index_t first_id = global_id * stride;
    index_t second_id = (global_id + ((len+1) / 2) ) * stride;

    data[first_id] = fmax(data[first_id], data[second_id]);
}

__kernel void vector_range (
    index_t length,
    __global index_t * data
) {
    index_t global_id = get_global_id (0);
    if (global_id >= 0 && global_id < length) {
        data[global_id] = global_id;
    }
}

__kernel void vector_imax_bang (
    __global double * data,
    const index_t len,
    const index_t stride,
    // Stores 
    __global index_t * indices
) {
    index_t global_id = get_global_id (0);
    index_t first_id = global_id * stride;
    index_t second_id = (global_id + ((len+1) / 2) ) * stride;
    double first_value = data[first_id];
    double second_value = data[second_id];
    index_t first_index = indices[first_id];
    index_t second_index = indices[second_id];

    if (first_value > second_value ||
            (first_value == second_value && indices[first_id] <= indices[second_id])) {
        // in case of equality, prefer the smaller idx
        // This block is suppose to execute the below, but it is actually redundant
        // Comments included for clarity sake
        // data[first_id] = first_value;
        // indices[first_id] = first_index;
    } else {
        data[first_id] = second_value;
        indices[first_id] = second_index;
    }
}

__kernel void vector_imin_bang (
    __global double * data,
    const index_t len,
    const index_t stride,
    // Stores 
    __global index_t * indices
) {
    index_t global_id = get_global_id (0);
    index_t first_id = global_id * stride;
    index_t second_id = (global_id + ((len+1) / 2) ) * stride;
    double first_value = data[first_id];
    double second_value = data[second_id];
    index_t first_index = indices[first_id];
    index_t second_index = indices[second_id];

    if (first_value < second_value ||
            (first_value == second_value && indices[first_id] <= indices[second_id])) {
        // in case of equality, prefer the smaller idx
        // This block is suppose to execute the below, but it is actually redundant
        // Comments included for clarity sake
        // data[first_id] = first_value;
        // indices[first_id] = first_index;
    } else {
        data[first_id] = second_value;
        indices[first_id] = second_index;
    }
}

__kernel void vector_min_bang (
    // The actual parameters to the kernel
    __global double * data,
    const index_t len,
    const index_t stride
) {
    index_t global_id = get_global_id (0);
    index_t first_id = global_id * stride;
    index_t second_id = (global_id + ((len+1) / 2) ) * stride;

    data[first_id] = fmin(data[first_id], data[second_id]);
}
#endif

