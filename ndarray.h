typedef struct {
    double * data;
    size_t * shape;
    size_t * strides;
    size_t ndims;
} ndarray;


unsigned ndarray_datasize(const ndarray *);

// addition
ndarray * ndarray_add(const ndarray *, ndarray *);
ndarray * ndarray_add_scalar(ndarray *, double);
void ndarray_add_bang(const ndarray *, ndarray *);

// multiplication
ndarray * ndarray_mul(ndarray*, ndarray*);


