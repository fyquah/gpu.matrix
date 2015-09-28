#include "types.h"

typedef struct {
    double * data;
    index_t * shape;
    index_t * strides;
    index_t ndims;
} ndarray;


unsigned ndarray_datasize(const ndarray *);

// helper functions
ndarray * ndarray_clone_structure(const ndarray *);
ndarray * ndarray_clone(const ndarray *);
unsigned ndarray_elements_count(const ndarray *);
ndarray * ndarray_coerce_stride(const ndarray *, index_t *);
ndarray * ndarray_broadcast(const ndarray *, index_t, index_t *);

// constructors
ndarray * ndarray_constructor(double *, index_t, index_t*, index_t*);
ndarray * ndarray_constructor_from_shape(index_t, index_t*);

// destructor
void ndarray_release(ndarray *);

bool ndarray_equals(const ndarray *, const ndarray*);
bool ndarray_equals_scalar(const ndarray*, const double);
double * ndarray_flatten(const ndarray *);

// getters and setters
void ndarray_set_1d(ndarray*, long i, double v);
void ndarray_set_2d(ndarray*, long i, long j, double v);
void ndarray_set_nd(ndarray*, const index_t * indexes, double v);

// add
ndarray * ndarray_add(const ndarray *, const ndarray *);
ndarray * ndarray_add_scalar(const ndarray *, const double);
void ndarray_add_bang(ndarray *, const ndarray *);
void ndarray_add_scalar_bang(ndarray *, const double);

// sub
ndarray * ndarray_sub(const ndarray *, const ndarray *);
ndarray * ndarray_sub_scalar(const ndarray *, const double);
void ndarray_sub_bang(ndarray *, const ndarray *);
void ndarray_sub_scalar_bang(ndarray *, const double);

// mul
ndarray * ndarray_mul(const ndarray *, const ndarray *);
ndarray * ndarray_mul_scalar(const ndarray *, const double);
void ndarray_mul_bang(ndarray *, const ndarray *);
void ndarray_mul_scalar_bang(ndarray *, const double);

// div
ndarray * ndarray_div(const ndarray *, const ndarray *);
ndarray * ndarray_div_scalar(const ndarray *, const double);
void ndarray_div_bang(ndarray *, const ndarray *);
void ndarray_div_scalar_bang(ndarray *, const double);

