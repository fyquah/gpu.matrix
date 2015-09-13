#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void add_scalar (__global const double * X, double y, __global double * Z) {
    int i = get_global_id (0);

    Z[i] = X[i] + y; 
}
