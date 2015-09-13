typedef struct {
    double * data;
    size_t * shape;
    size_t * strides;
    size_t ndims;
} ndarray;


unsigned ndarray_datasize(const ndarray *);

// helper functions
ndarray * ndarray_clone_structure(const ndarray *);
unsigned ndarray_elements_count(const ndarray *);

// addition
ndarray * ndarray_add(const ndarray *, const ndarray *);
ndarray * ndarray_add_scalar(const ndarray *, double);
void ndarray_add_bang(ndarray *, const ndarray *);

// multiplication
ndarray * ndarray_mul(ndarray*, ndarray*);


