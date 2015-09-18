#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void add_scalar_bang (__global double * X, double y) {
    int i = get_global_id (0);

    X[i] = X[i] + y; 
}
