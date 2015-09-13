#include <CL/cl.h>

size_t * copy (size_t * arr, size_t len) {
    size_t * copy = (size_t*) malloc(sizeof(size_t) * len);

    for (size_t i = 0 ; i < len ; i++) {
        copy[i] = arr[i];
    }

    return copy;
}
