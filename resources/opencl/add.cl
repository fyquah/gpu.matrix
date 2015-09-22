#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// vector addition : add, strides is definitely equals to one

struct ndarray {
    double * data;
    unsigned * shape;
    unsigned * strides;
    unsigned ndims;
};
typedef struct ndarray ndarray;
typedef unsigned index_t;

// multipurpose add kernel
// no local memory fetching optmiziations will be done
// currently assumes that both NDArray have same dimensions
__kernel void add_vectors (
    __global const double * data_x,
    __global const double * data_y,
    __global double * data_output
) {
    index_t i = get_global_id (0);
    data_output[i] = data_x[i] + data_y[i];
}

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
    index_t ndims = max(ndims_x, ndims_y);
    
    switch (ndims) {
        case 0: {
            *data_output = *data_x + *data_y;
            break;
        }
        case 1: {
            index_t i = get_global_id (0);
            data_output[i] = data_x[i] + data_y[i];
            break;
        }
        case 2:
        case 3: {
            index_t idx_x = 0, idx_y = 0;
            for (int i = 0 ; i < ndims ; i++) {
                index_t global_id = (index_t) get_global_id (i);
                idx_x += strides_x[i] * global_id; 
                idx_y += strides_y[i] * global_id; 
            }
            data_output[idx_x] = data_x[idx_x] + data_y[idx_y];
            break;
        }
        default: {
            // assumed that both NDArray object has been coerced to relevant form
            // a vector addition will be done
            add_vectors (data_x, data_y, data_output);
            break;
        }
    } 
}
