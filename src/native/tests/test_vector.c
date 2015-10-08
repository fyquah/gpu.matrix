#include "test_vector.h"

// because 2^10 is kinda cool :
#define LENGTH 1025 
#define TEST  
#define INIT_TEST_VECTOR(v) \
    vector * v = malloc(sizeof(vector)); \
    v->data = malloc(sizeof(double) * LENGTH); \
    v->length = LENGTH; \
    v->stride = 1; \
    for (unsigned i = 0 ; i < LENGTH ; i++) { \
        v->data[i] = ((double) ((rand() % LENGTH) - LENGTH / 2)) * 2.47840239; \
    }

#define FREE_TEST_VECTOR(v) \
    free(v->data); \
    free(v);

#define TEST_INIT \
    INIT_TEST_VECTOR(v_x); \
    INIT_TEST_VECTOR(v_y);

#define TEST_INIT_WITH_COPY \
    TEST_INIT; \
    vector *copy_v_x = gpu_matrix_vector_copy(v_x); \
    vector *copy_v_y = gpu_matrix_vector_copy(v_y);

#define TEST_FREE \
    FREE_TEST_VECTOR(v_x); \
    FREE_TEST_VECTOR(v_y);

#define TEST_FREE_WITH_COPY \
    FREE_TEST_VECTOR(copy_v_x); \
    FREE_TEST_VECTOR(copy_v_y);

static inline double at(vector * v, index_t idx) {
    return v->data[idx * v->stride];
}

static double inline min(double a, double b) {
    return (a < b ? a : b);
}

static double inline max(double a, double b) {
    return (a > b ? a : b);
}

static bool is_similiar(double a, double b) {
    if (a < b) {
        double tmp_a = a;
        a = b;
        b = tmp_a;
    }

    if (a == b || fabs((a-b)/a) <= 0.001) {
        return true;
    } else {
        return false;
    }
    return false;
}

DEFTEST(vector_copy) {
    TEST_INIT;
    vector * copy_x = gpu_matrix_vector_copy(v_x);
    ASSERT(copy_x->length == v_x->length);
    for (index_t i = 0 ; i < v_x->length ; i++) {
        ASSERT(copy_x->data[i * copy_x->stride] == v_x->data[i * v_x->stride]);
    }
    free(copy_x->data);
    free(copy_x);
    TEST_FREE;
}

DEFTEST(vector_dot_product) {
    TEST_INIT;
    double dot_product, expected_dot_product;
    dot_product = gpu_matrix_vector_dot(v_x, v_y);
    expected_dot_product = 0;
    for (index_t i = 0 ; i < v_x->length ; i++) {
        expected_dot_product += v_x->data[i] * v_y->data[i]; 
    }
    ASSERT(dot_product == expected_dot_product);

    TEST_FREE;
}

DEFTEST(vector_axpy) {
    TEST_INIT_WITH_COPY;
    gpu_matrix_vector_axpy(copy_v_x, 12.0, copy_v_y);
    ASSERT(copy_v_x->length == v_x->length);
    for (index_t i = 0 ; i < v_x->length ; i++) {
        ASSERT(
            is_similiar(
                v_x->data[i * v_x->stride] * 12 + v_y->data[i * v_y->stride],
                copy_v_x->data[i * copy_v_x->stride]
            )
        );
    }
    TEST_FREE_WITH_COPY;
}

DEFTEST(vector_scal) {
    TEST_INIT_WITH_COPY;
    const double factor = 2.7817;
    gpu_matrix_vector_scal(copy_v_x, factor);
    ASSERT(copy_v_x->length == v_x->length);
    for (index_t i = 0 ; i < v_x->length ; i++) {
        ASSERT(
            is_similiar(
                v_x->data[i * v_x->stride] * factor,
                copy_v_x->data[i * copy_v_x->stride]
            )
        );
    }
    TEST_FREE_WITH_COPY;
}

DEFTEST(vector_swap) {
    TEST_INIT_WITH_COPY;
    gpu_matrix_vector_swap(copy_v_x, copy_v_y);
    ASSERT(copy_v_x->length == v_y->length);
    ASSERT(copy_v_y->length == v_x->length);
    for (index_t i = 0 ; i < v_x->length ; i++) {
        ASSERT(
            is_similiar(at(v_x, i), at(copy_v_y, i))
        );
        ASSERT(
            is_similiar(at(v_y, i), at(copy_v_x, i))
        );
    }
    TEST_FREE_WITH_COPY;
}

DEFTEST(vector_rot) {
    TEST_INIT_WITH_COPY;
    double c = 23.0, s = 249.0;
    gpu_matrix_vector_rot(copy_v_x, copy_v_y, c, s);
    ASSERT(v_x->length == copy_v_x->length);
    for(index_t i = 0 ; i < v_x->length ; i++) {
        ASSERT(
            is_similiar(at(copy_v_x, i), at(v_x, i) * c + s * at(v_y, i))
        );
    }
    TEST_FREE_WITH_COPY;
}

DEFTEST(vector_nrm2) {
    TEST_INIT;
    double output = gpu_matrix_vector_nrm2(v_x);
    double expected = 0.0;
    for (index_t i = 0 ; i < v_x->length ; i++) {
        expected += at(v_x, i) * at(v_x, i);
    }
    ASSERT(is_similiar(expected, output));
}

DEFTEST(vector_asum) {
    TEST_INIT;
    double output = gpu_matrix_vector_asum(v_x);
    double expected = 0.0;
    for (index_t i = 0 ; i < v_x->length ; i++) {
        expected += fabs(at(v_x, i));
    }
    ASSERT(is_similiar(expected, output));
}

DEFTEST(vector_amax) {
    TEST_INIT;
    double output = gpu_matrix_vector_amax(v_x);
    if (v_x->length == 0) return;
    double expected = fabs(at(v_x, 0));
    for (index_t i = 1 ; i < v_x->length ; i++) {
        expected = max(fabs(at(v_x, i)), expected);
    }
    ASSERT(is_similiar(expected, output));
}

DEFTEST(vector_amin) {
    TEST_INIT;
    double output = gpu_matrix_vector_amin(v_x);
    if (v_x->length == 0) return;
    double expected = fabs(at(v_x, 0));
    for (index_t i = 1 ; i < v_x->length ; i++) {
        expected = min(fabs(at(v_x, i)), expected);
    }
    ASSERT(is_similiar(expected, output));
}

DEFTEST(vector_imin) {
    TEST_INIT;
    double output = gpu_matrix_vector_imin(v_x);
    if (v_x->length == 0) return;

    index_t best_index = 0;
    double best_value = at(v_x, 0);
    for (index_t i = 1 ; i < v_x->length ; i++) {
        if (at(v_x, i) < best_value) {
            best_index = i;
            best_value = at(v_x, i);
        }
    }
    ASSERT(is_similiar(output, best_index));
}

DEFTEST(vector_imax) {
    TEST_INIT;
    double output = gpu_matrix_vector_imax(v_x);
    if (v_x->length == 0) return;

    index_t best_index = 0;
    double best_value = at(v_x, 0);
    for (index_t i = 1 ; i < v_x->length ; i++) {
        if (at(v_x, i) > best_value) {
            best_index = i;
            best_value = at(v_x, i);
        }
    }
    ASSERT(is_similiar(output, best_index));
}

DEFTEST(vector_iamin) {
    TEST_INIT;
    double output = gpu_matrix_vector_iamin(v_x);
    if (v_x->length == 0) return;

    index_t best_index = 0;
    double best_value = fabs(at(v_x, 0));
    for (index_t i = 1 ; i < v_x->length ; i++) {
        if (fabs(at(v_x, i)) < best_value) {
            best_index = i;
            best_value = fabs(at(v_x, i));
        }
    }
    ASSERT(is_similiar(output, best_index));
}

DEFTEST(vector_iamax) {
    TEST_INIT;
    double output = gpu_matrix_vector_iamax(v_x);
    if (v_x->length == 0) return;

    index_t best_index = 0;
    double best_value = fabs(at(v_x, 0));
    for (index_t i = 1 ; i < v_x->length ; i++) {
        if (fabs(at(v_x, i)) > best_value) {
            best_index = i;
            best_value = fabs(at(v_x, i));
        }
    }
    ASSERT(is_similiar(output, best_index));
}

DEFTEST(vector_add) {
    TEST_INIT_WITH_COPY;
    vector * arr_v[3];
    vector * output;

    arr_v[0] = v_x;
    arr_v[1] = v_y;
    arr_v[2] = v_x;

    output = gpu_matrix_vector_add_arbitary(3, arr_v);
    ASSERT(v_x->length == output->length);
    for (index_t i = 0 ; i < v_x->length ; i++) {
        ASSERT(is_similiar(
                    at(v_x, i) + at(v_x, i) + at(v_y, i), at(output, i)
        ));
    }
    free(output->data);
    free(output);

    output = gpu_matrix_vector_add_2(v_x, v_y);
    for (index_t i = 0 ; i < v_x->length ; i++) {
        ASSERT(is_similiar(
                    at(v_x, i) + at(v_y, i), at(output, i)
        ));
    }
    free(output->data);
    free(output);

    output = gpu_matrix_vector_add_scalar(v_x, 23.0);
    for(index_t i = 0 ; i < v_x->length ; i++) {
        ASSERT(is_similiar(
                    at(v_x, i) + 23.0, at(output, i)
        ));
    }
    free(output->data);
    free(output);

    TEST_FREE_WITH_COPY;
}

void test_vector() {
    RUNTEST(vector_copy);
    RUNTEST(vector_axpy);
    RUNTEST(vector_scal);
    RUNTEST(vector_swap);
    RUNTEST(vector_rot);
    RUNTEST(vector_dot_product);
    RUNTEST(vector_nrm2);
    RUNTEST(vector_asum);
    RUNTEST(vector_amax);
    RUNTEST(vector_amin);
    RUNTEST(vector_imin);
    RUNTEST(vector_imax);
    RUNTEST(vector_iamin);
    RUNTEST(vector_iamax);
    RUNTEST(vector_add);
}

