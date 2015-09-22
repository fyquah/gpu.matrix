#include "array.h"

#define COPY_FACTORY(T) T * array_##T##_copy(T * arr, unsigned long long len) { \
    T * copy = (T*) malloc(sizeof(T) * len);\
    for (unsigned long i = 0; i < len ; ++i) { \
        copy[i] = arr[i]; \
    }\
    return copy;\
}\

#define IS_EQUAL_FACTORY(T) bool array_##T##_is_equal(T * arr_x, T * arr_y, unsigned long long len){ \
    for (unsigned long long i = 0 ; i < len ; i++) { \
        if (arr_x[i] != arr_y[i]) { \
            return 0; \
        } \
    } \
}

COPY_FACTORY(size_t);
COPY_FACTORY(index_t);
COPY_FACTORY(double);

IS_EQUAL_FACTORY(index_t);
