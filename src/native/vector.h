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
vector_buffer gpu_matrix_vector_to_vector_buffer(vector *, cl_mem buffer);
vector * gpu_matrix_vector_copy(vector * x);

// (Partial) BLAS 1 headers. Note: BLAS headers mutate the argument vectors, they will
// never return an argument!
extern void gpu_matrix_vector_axpy(vector * x, double a, vector * y);
extern void gpu_matrix_vector_scal(vector * x, double a);
extern void gpu_matrix_vector_swap(vector * x, vector * y);
extern void gpu_matrix_vector_rot(vector * x, vector * y, double, double);
extern double gpu_matrix_vector_dot(vector * x, vector * y);
extern double gpu_matrix_vector_nrm2(vector * x);
extern double gpu_matrix_vector_asum(vector * x);
extern double gpu_matrix_vector_amax(vector*);
extern double gpu_matrix_vector_amin(vector*);
extern index_t gpu_matrix_vector_imin(vector * x);
extern index_t gpu_matrix_vector_imax(vector * x);
extern index_t gpu_matrix_vector_iamin(vector * x);
extern index_t gpu_matrix_vector_iamax(vector * x);
extern bool gpu_matrix_vector_value_equality(vector * v_x, vector * v_y);
extern bool gpu_matrix_vector_value_equality_ndarray(vector * v_x, ndarray * arr_y);

// Arimethic functions
#define VECTOR_ARIMETHIC_HEADER_FACTORY(name) \
vector * gpu_matrix_vector_##name##_scalar(vector *, double); \
vector * gpu_matrix_vector_##name##_2(vector *, vector *); \
vector * gpu_matrix_vector_##name##_arbitary(unsigned, vector *[]);

VECTOR_ARIMETHIC_HEADER_FACTORY(add);
VECTOR_ARIMETHIC_HEADER_FACTORY(mul);
VECTOR_ARIMETHIC_HEADER_FACTORY(sub);
VECTOR_ARIMETHIC_HEADER_FACTORY(div);

// element map functions (more like utilities then actually useful)
vector * gpu_matrix_vector_pow(vector *, double);

// element reduce functions
double gpu_matrix_vector_sum(vector *);

#endif
