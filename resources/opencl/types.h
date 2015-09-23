#pragma OPENCL EXTENSION cl_khr_fp64 : enable
struct ndarray {
    double * data;
    unsigned * shape;
    unsigned * strides;
    unsigned ndims;
};
typedef struct ndarray ndarray;
typedef unsigned index_t;
