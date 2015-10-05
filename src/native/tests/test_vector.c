#include "../utils.h"
#include "../vector.h"
#include "../types.h"
#include "test_vector.h"

// because 2^22 is kinda cool :
#define LENGTH 4194304

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
    int d = 123;
    double *data_a, *data_b;
    bool flag;
    vector * axpy_output, expected_axpy_output; 
    double dot_product_output, expected_dot_product_output;
    double asum_output, expected_asum_output;

    // Initialize and prepare test data
    data_a = malloc(sizeof(double) * LENGTH);
    data_b = malloc(sizeof(double) * LENGTH);
    a.length = LENGTH;
    b.length = LENGTH;
    a.stride = 1;
    b.stride = 1;
    a.data = data_a;
    b.data = data_b;

    for (int i = 0 ; i < LENGTH ; i++) {
        a.data[i] = i * 1.2;
        b.data[i] = i * 1.3;
    }
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
        expected_asum_output += a.data[i];
    }

    if(is_simliar(asum_output, expected_asum_output)) {
        puts("CORRECT!\n");
    } else {
        puts("WRONG!\n");
    }

}

void test_vector() {
    test_vector_blas();
}
