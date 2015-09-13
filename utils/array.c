#include <CL/cl.h>

#define COPY_FACTORY(T) T * array_##T##_copy(T * arr, unsigned long long len) { \
    T * copy = (T*) malloc(sizeof(T) * len);\
    for (unsigned long i = 0; i < len ; ++i) { \
        copy[i] = arr[i]; \
    }\
    return copy;\
}\

COPY_FACTORY(size_t);
