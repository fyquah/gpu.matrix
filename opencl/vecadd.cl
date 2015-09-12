__kernel void vecadd (__global const int * A, __global const int * B, __global int * C) {
    int idx = get_global_id (0);

    C[idx] = A[idx] + B[idx];

}

