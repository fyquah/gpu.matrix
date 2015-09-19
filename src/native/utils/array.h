#define COPY_HEADER_FACTORY(T) T * array_##T##_copy(T * arr, unsigned long long len);

COPY_HEADER_FACTORY(size_t);
