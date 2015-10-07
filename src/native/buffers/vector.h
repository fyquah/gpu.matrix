#ifndef BUFFERS_VECTOR_H
#define BUFFERS_VECTOR_H

#include <time.h>
#include "../types.h"
#include "../utils.h"

// BANG functions modify the first argument!

void gpu_matrix_vector_buffer_axpy_BANG(vector_buffer *, double, vector_buffer *, cl_command_queue);
void gpu_matrix_vector_buffer_asum_BANG(vector_buffer *, cl_command_queue);
void gpu_matrix_vector_buffer_mul_BANG(vector_buffer *, vector_buffer *, cl_command_queue);
void gpu_matrix_vector_buffer_square_BANG(vector_buffer *, cl_command_queue);
void gpu_matrix_vector_buffer_rot_BANG(vector_buffer *, vector_buffer*, double, double, cl_command_queue);

#endif
