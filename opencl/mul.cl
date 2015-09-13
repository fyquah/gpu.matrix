__kernel void mul (
    __global int * m_x,
    __global int * m_y,
    unsigned width_x,
    unsigned width_y,
    __global int * m_z
) {
    // note : width refers to number of columns 
    unsigned row = get_global_id (0);
    unsigned col = get_global_id (1);

    int sum = 0.0;

    for (int i = 0 ; i < width_x ; i++) {
        sum += m_x[row*width_x +i] * m_y[i*width_y+col];  
    }

    m_z[row*width_y + col] = sum; 
}
