#ifndef BUFFERS_VECTOR_H
#define BUFFERS_VECTOR_H

#include <time.h>
#include "../types.h"
#include "../utils.h"

// BANG functions modify the first argument!

void gpu_matrix_vector_buffer_axpy_BANG(vector_buffer *, double, vector_buffer *, cl_command_queue);

#endif
