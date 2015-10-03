#include "types.h"

typedef struct {
    double * data;
    index_t length;
    index_t stride;
} vector;

// BLAS reference:
// http://www.netlib.org/lapack/lug/node145.html
//
// _ROTG Generate plane rotation S, D
// _ROTMG    Generate modified plane rotation    S, D
// _ROT  Apply plane rotation    S, D
// _ROTM Apply modified plane rotation   S, D
// DOTU  $ dot \leftarrow x ^ {T} y $   C, Z
// _DOTC     $ dot \leftarrow x ^ {H} y $   C, Z

vector * gpu_matrix_vector_axpy(vector * x, double a, vector * y);
vector * gpu_matrix_vector_swap(vector * x, vector * y);
vector * gpu_matrix_vector_scal(vecetor * x, double a);
vector * gpu_matrix_vector_copy();
double gpu_matrix_vector_dot(vector * x, vector * y);
double gpu_matrix_vector_nrm2(vector * x);
double gpu_matrix_vector_asum(vector * x);

