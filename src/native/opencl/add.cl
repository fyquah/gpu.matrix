#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// vector addition : add, strides is definitely equals to one

struct ndarray {
    double * data;
    unsigned * shape;
    unsigned * strides;
    unsigned ndims;
};
typedef struct ndarray ndarray;
typedef unsigned long index_t;

ndarray ndarray_construct_g (
    double * data,
    unsigned * shape,
    unsigned * strides,
    unsigned ndims) {

    ndarray output;
    output.data = data;
    output.shape = shape;
    output.strides = strides;
    output.ndims = ndims;
    return output;
}

// multipurpose add kernel
// no local memory fetching optmiziations will be done
__kernel void add(
    __global const double * data_x,
    __global const index_t * shape_x,
    __global const index_t * strides_x,
    index_t ndims_x,
    __global const double * data_y,
    __global const index_t * shape_y,
    __global const index_t * strides_y,
    index_t ndims_y, 
    __global double * data_output
) {
    int ndims = 2; 

    index_t idx_x = 0, idx_y = 0;
    for  (int i = 0 ; i < ndims ; i++) {
        index_t global_id = (index_t) get_global_id (i);
        idx_x += strides_x[i] * global_id; 
        idx_y += strides_y[i] * global_id; 
    }
    data_output[idx_x] = data_x[idx_x] + data_y[idx_y];
}
// A and B are input matrices
// strides is a stride that describe the RELATIVE STRIDE of matrix B w.r.t to A
// row is the localBuffer for a single row
// C is the output matrix
__kernel void add_matrix (
    __global const ndarray * A,
    __global const ndarray * B,
    __local double * row_A,
    __local double * row_B,
    __global const ndarray * C) {

    // assume both A and B has ndims of 2
    unsigned height = A->shape[0];
    unsigned width = A->shape[1];
    unsigned i = get_global_id(0);
    unsigned j = get_local_id(0);

}
