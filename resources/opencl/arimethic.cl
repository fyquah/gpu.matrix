#pragma OPENCL EXTENSION cl_khr_fp64 : enable

struct ndarray {
    double * data;
    unsigned * shape;
    unsigned * strides;
    unsigned ndims;
};
typedef struct ndarray ndarray;
typedef unsigned index_t;

// multipurpose arimethic kernels
// no local memory fetching optmiziations will be done
// there is specialized ops for vectors (there should be for matrices too)

#define ARIMETHIC_FACTORY(op_name, op) __kernel void op_name##_scalar ( \
    __global const double * X, \
    double y, \
    __global double * Z) { \
    int i = get_global_id (0); \
\
    Z[i] = X[i] op y; \
} \
\
__kernel void op_name##_vectors (\
    __global const double * data_x,\
    __global const double * data_y,\
    __global double * data_output\
) {\
    index_t i = get_global_id (0);\
    data_output[i] = data_x[i] op data_y[i];\
}\
\
__kernel void op_name(\
    __global const double * data_x,\
    __global const index_t * shape_x,\
    __global const index_t * strides_x,\
    index_t ndims_x,\
    __global const double * data_y,\
    __global const index_t * shape_y,\
    __global const index_t * strides_y,\
    index_t ndims_y, \
    __global double * data_output\
) {\
    index_t ndims = max(ndims_x, ndims_y);\
    \
    switch (ndims) {\
        case 0: {\
            *data_output = *data_x op *data_y;\
            break;\
        }\
        case 1: {\
            index_t i = get_global_id (0);\
            data_output[i] = data_x[i] op data_y[i];\
            break;\
        }\
        case 2:\
        case 3: { \
            index_t idx_x = 0, idx_y = 0;\
            for (int i = 0 ; i < ndims ; i++) {\
                index_t global_id = (index_t) get_global_id (i);\
                idx_x += strides_x[i] * global_id; \
                idx_y += strides_y[i] * global_id; \
            }\
            data_output[idx_x] = data_x[idx_x] op data_y[idx_y];\
            break;\
        } \
        default: { \
            op_name##_vectors (data_x, data_y, data_output);\
            break;\
        } \
    } \
}

ARIMETHIC_FACTORY(add, +)
ARIMETHIC_FACTORY(sub, -)
ARIMETHIC_FACTORY(mul, *)
ARIMETHIC_FACTORY(div, /)

