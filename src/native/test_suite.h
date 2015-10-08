// A little working test suite

#define DEFTESTCLASS(name)  \
    void _test_class_##name()

#define DEFTEST(name) \
    void _test_##name(bool * _flag, unsigned * _assertions_count, unsigned * _line)

#define ASSERT(condition) \
    if (!condition) { \
        *_flag = false; \
        *_line = __LINE__; \
        return; \
    } else { \
        *_assertions_count = *_assertions_count + 1; \
    }

#define RUNTEST(NAME) \
    { \
        srand(time(NULL)); \
        bool flag = true; \
        unsigned line; \
        unsigned assertions_count = 0; \
        const char test_name[] = #NAME; \
        _test_##NAME(&flag, &assertions_count, &line); \
\
        puts("======================="); \
        puts("Test " #NAME); \
        printf("%u assertion(s) ", assertions_count); \
        if(flag) { \
            printf("\n"); \
        } else { \
            printf("with a failure at line %u\n", line); \
        } \
        puts("=======================\n"); \
    }

