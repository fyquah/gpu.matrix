#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void vecadd (__global const double * A, __global const double * B, __global double * C) {
    int idx = get_global_id (0);

    C[idx] = A[idx] + B[idx];

}
