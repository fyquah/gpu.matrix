// BLAS 1 vector functions and subroutines
// BLAS reference:
// http://www.netlib.org/lapack/lug/node145.html
//
// _ROTG Generate plane rotation S, D
// _ROTMG    Generate modified plane rotation    S, D
// _ROT  Apply plane rotation    S, D
// _ROTM Apply modified plane rotation   S, D
// DOTU  $ dot \leftarrow x ^ {T} y $   C, Z
// _DOTC     $ dot \leftarrow x ^ {H} y $   C, Z

#ifndef VECTOR_H
#define VECTOR_H

#include <time.h>
#include "types.h"
#include "utils.h"
#include "buffers/vector.h"

#define THREADS_COUNT 100;

// Release memory
// Does not naively release memory, need to do some GC resolution
void gpu_matrix_release(vector *);

// BLAS 1 headers
vector * gpu_matrix_vector_axpy(vector * x, double a, vector * y);
vector * gpu_matrix_vector_scal(vector * x, double a);
vector * gpu_matrix_vector_copy(vector * x);
void gpu_matrix_vector_swap(vector * x, vector * y);
double gpu_matrix_vector_dot(vector * x, vector * y);
double gpu_matrix_vector_nrm2(vector * x);
double gpu_matrix_vector_asum(vector * x);

#endif
