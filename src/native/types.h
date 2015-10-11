#ifndef TYPE_H
#define TYPE_H
#include <stdbool.h>
#include <stdint.h>
#include "libs/cl.h"

// TODO : How to support 64 bit machines?
typedef uint32_t index_t;
typedef unsigned long long ull;
typedef const unsigned kernel_type_t;

typedef struct {
    double * data;
    index_t length;
    index_t stride;
} vector;

typedef struct {
    cl_mem buffer;
    index_t length;
    index_t stride;
    size_t datasize;
} vector_buffer;

typedef struct {
    double * data;
    index_t * shape;
    index_t * strides;
    index_t ndims;
} ndarray;

// is this useful?
typedef struct {
    cl_mem * data;
} scalar_buffer;

#endif
