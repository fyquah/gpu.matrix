#ifndef BUFFERS_VECTOR_H
#define BUFFERS_VECTOR_H

#include <time.h>
#include "../types.h"
#include "../utils.h"

// BANG functions modify the first argument!

// ========== BLAS LEVEL 1 =============

// map functions
extern void gpu_matrix_vector_buffer_axpy_BANG(vector_buffer *,vector_buffer *, double, cl_command_queue);
extern void gpu_matrix_vector_buffer_scal_BANG(vector_buffer *, double alpha, cl_command_queue);
extern void gpu_matrix_vector_buffer_rot_BANG(vector_buffer *, vector_buffer*, double, double, cl_command_queue);
extern void gpu_matrix_vector_buffer_abs_BANG(vector_buffer *, cl_command_queue);
extern void gpu_matrix_vector_buffer_square_BANG(vector_buffer *, cl_command_queue);

// reduce functions
extern void gpu_matrix_vector_buffer_asum_BANG(vector_buffer *, cl_command_queue);
extern void gpu_matrix_vector_buffer_min_BANG(vector_buffer *, cl_command_queue);
extern void gpu_matrix_vector_buffer_max_BANG(vector_buffer *, cl_command_queue);
extern cl_mem gpu_matrix_vector_buffer_imin(vector_buffer *, cl_command_queue);
extern cl_mem gpu_matrix_vector_buffer_imax(vector_buffer *, cl_command_queue);

// ========== END OF BLAS LEVEL 1 =============

// common arimethic ops
extern void gpu_matrix_vector_buffer_add_BANG(vector_buffer *, vector_buffer*, cl_command_queue);
extern void gpu_matrix_vector_buffer_add_scalar_BANG(vector_buffer *, double, cl_command_queue);
extern void gpu_matrix_vector_buffer_sub_BANG(vector_buffer *, vector_buffer*, cl_command_queue);
extern void gpu_matrix_vector_buffer_mul_BANG(vector_buffer *, vector_buffer*, cl_command_queue);
extern void gpu_matrix_vector_buffer_div_BANG(vector_buffer *, vector_buffer*, cl_command_queue);

#endif
