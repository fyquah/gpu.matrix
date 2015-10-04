#ifndef VECTOR_CL
#define VECTOR_CL

#include <types.h>

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

__kernel void vector_axpy (
    // The actual parameters to the kernel
    __global const double * data_x,
    const index_t length_x,
    const index_t stride_x,
    const double alpha,
    __global const double * data_y,
    const index_t length_y,
    const index_t stride_y,
    // Local buffers for caching
    __local double * local_cache_x,
    __local double * local_cache_y,
    // Output memory locations
    __global double * data_output
) {
    index_t global_id = get_global_id (0);
    index_t local_id = get_local_id (0);

    local_cache_x[local_id] = data_x[global_id * stride_x];
    local_cache_y[local_id] = data_y[global_id * stride_y];

    data_output[global_id] = alpha * 
        local_cache_x[local_id] +
        local_cache_y[local_id];
}

#endif
