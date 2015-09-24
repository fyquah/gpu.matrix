#include "../types.h"
#include "../libs/cl.h"

#define COPY_HEADER_FACTORY(T) T * array_##T##_copy(T * arr, unsigned long long len);
#define IS_EQUAL_HEADER_FACTORY(T) bool array_##T##_IS_EQUAL(T *, T *, unsigned long long);

COPY_HEADER_FACTORY(size_t);
COPY_HEADER_FACTORY(index_t);
COPY_HEADER_FACTORY(double);

IS_EQUAL_HEADER_FACTORY(index_t);
bool array_index_t_is_equal(index_t*, index_t*, unsigned long long);
