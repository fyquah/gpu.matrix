#include "../utils.h"
#include "../vector.h"
#include "../types.h"
#include "test_vector.h"

// because 2^22 is kinda cool :
#define LENGTH 4194304

static double inline min(double a, double b) {
    return (a < b ? a : b);
}

static double inline max(double a, double b) {
    return (a > b ? a : b);
}

bool is_simliar(double a, double b) {
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

void test_vector_blas() {
    vector a, b;
    vector * copy_a, * copy_b;
    int d = 123;
    double *data_a, *data_b;
    double c, s;
    bool flag;
    vector * axpy_output, expected_axpy_output; 
    double dot_product_output, expected_dot_product_output;
    double asum_output, expected_asum_output;
    double nrm2_output, expected_nrm2_output;
    double amax_output, expected_amax_output;
    double amin_output, expected_amin_output;

    // Initialize and prepare test data
    data_a = malloc(sizeof(double) * LENGTH);
    data_b = malloc(sizeof(double) * LENGTH);
    a.length = LENGTH;
    b.length = LENGTH;
    a.stride = 1;
    b.stride = 1;
    a.data = data_a;
    b.data = data_b;
    c = 23.0;
    s = 49.440;

    for (int i = 0 ; i < LENGTH ; i++) {
        a.data[i] = (i - LENGTH / 2) * 1.2 + 1;
        b.data[i] = (i - LENGTH / 2) * 1.3 + 1;
    }

    copy_a = gpu_matrix_vector_copy(&a);
    copy_b = gpu_matrix_vector_copy(&b);
    // end of initialization

    // ----------------------------------
    // BLAS Level 1 tests
    // ----------------------------------
    
    puts("AXPY TEST:");
    axpy_output = gpu_matrix_vector_axpy(&a, 12.0, &b);
    flag = true;
    for (index_t i = 0 ; i < LENGTH ; i++) {
        if (fabs(a.data[i] * 12 + b.data[i] - axpy_output->data[i]) > 0.001) {
            flag = false;
            break;
        }
    }
    puts(flag ? "CORRECT!\n" : "WRONG!\n");

    puts("VECTOR_DOT TEST:");
    flag = true;
    dot_product_output = gpu_matrix_vector_dot(&a, &b);
    expected_dot_product_output = 0.0;
    for (index_t i = 0;  i < LENGTH ; i++) {
        expected_dot_product_output += a.data[i] * b.data[i];
    }

    if (fabs(expected_dot_product_output - dot_product_output) / 
            expected_dot_product_output <= 0.001) {
        puts("CORRECT!\n");
    } else {
        puts("WRONG!\n");
    }

    puts("VECTOR_ASUM TEST:");
    flag = true;
    asum_output = gpu_matrix_vector_asum(&a);
    expected_asum_output = 0;
    for (index_t i = 0 ; i < LENGTH ; i++) {
        expected_asum_output += fabs(a.data[i]);
    }

    if(is_simliar(asum_output, expected_asum_output)) {
        puts("CORRECT!\n");
    } else {
        puts("WRONG!\n");
    }

    puts("VECTOR_NRM2 TEST:");
    flag = true;
    nrm2_output = gpu_matrix_vector_nrm2(&a);
    expected_nrm2_output = 0;
    for (index_t i = 0 ; i < LENGTH ; i++) {
        expected_nrm2_output += a.data[i] * a.data[i];
    }

    if(is_simliar(nrm2_output, expected_nrm2_output)) {
        puts("CORRECT!\n");
    } else {
        puts("WRONG!\n");
    }

    puts("VECTOR_ROT TEST:");
    flag = true;
    gpu_matrix_vector_rot(copy_a, copy_b, c, s);
    for (index_t i = 0 ; i < LENGTH ; i++) {
        if (!is_simliar(copy_a->data[i], c * a.data[i] + s * b.data[i]) &&
            !is_simliar(copy_b->data[i], c * a.data[i] - s * b.data[i])) {
            flag = false;
            break;
        }
    }

    puts(flag ? "CORRECT!\n" : "WRONG!\n");

    puts("VECTOR_AMIN TEST:");
    amin_output = gpu_matrix_vector_amin(&a);
    if (LENGTH == 0) {
        flag = true;
    } else {
        expected_amin_output = fabs(a.data[0]);
        for (index_t i = 1 ; i < LENGTH ; i++) {
            double v = fabs(a.data[i]);
            expected_amin_output = min(
                expected_amin_output,
                v
            );
        }

        flag = is_simliar(amin_output, expected_amin_output);
    }
    puts(flag ? "CORRECT!\n" : "WRONG!\n");

    puts("VECTOR_AMAX TEST:");
    amax_output = gpu_matrix_vector_amax(&a);
    if (LENGTH == 0) {
        flag = true;
    } else {
        expected_amax_output = fabs(a.data[0]);
        for (index_t i = 1 ; i < LENGTH ; i++) {
            double v = fabs(a.data[i]);
            expected_amin_output = max(
                expected_amax_output,
                v
            );
        }

        flag = is_simliar(amax_output, expected_amax_output);
    }
    puts(flag ? "CORRECT!\n" : "WRONG!\n");
}

void test_vector() {
    test_vector_blas();
}

