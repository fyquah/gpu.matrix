#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void add_bang (__global double * A, __global const double * B) {
    int idx = get_global_id (0);

    A[idx] = A[idx] + B[idx];

}
