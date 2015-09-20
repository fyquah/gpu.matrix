typedef struct {
    double * data;
    size_t * shape;
    size_t * strides;
    size_t ndims;
} ndarray;


unsigned ndarray_datasize(const ndarray *);

// helper functions
ndarray * ndarray_clone_structure(const ndarray *);
ndarray * ndarray_clone(const ndarray *);
unsigned ndarray_elements_count(const ndarray *);

// constructors
ndarray * ndarray_constructor(double *, size_t, size_t*, size_t*);
ndarray * ndarray_constructor_from_shape(size_t, size_t*);

// destructor
void ndarray_release(ndarray *);

// getters and setters
void ndarray_set_1d(ndarray*, long i, double v);
void ndarray_set_2d(ndarray*, long i, long j, double v);
void ndarray_set_nd(ndarray*, const long * indexes, double v);

// add
ndarray * ndarray_add(const ndarray *, const ndarray *);
ndarray * ndarray_add_scalar(const ndarray *, const double);
void ndarray_addbang(ndarray *, const ndarray *);
void ndarray_add_scalar_bang(ndarray *, const double);

// sub
ndarray * ndarray_sub(const ndarray *, const ndarray *);
ndarray * ndarray_sub_scalar(const ndarray *, const double);
void ndarray_subbang(ndarray *, const ndarray *);
void ndarray_sub_scalar_bang(ndarray *, const double);

// mul
ndarray * ndarray_mul(const ndarray *, const ndarray *);
ndarray * ndarray_mul_scalar(const ndarray *, const double);
void ndarray_mulbang(ndarray *, const ndarray *);
void ndarray_mul_scalar_bang(ndarray *, const double);

// div
ndarray * ndarray_div(const ndarray *, const ndarray *);
ndarray * ndarray_div_scalar(const ndarray *, const double);
void ndarray_divbang(ndarray *, const ndarray *);
void ndarray_div_scalar_bang(ndarray *, const double);

