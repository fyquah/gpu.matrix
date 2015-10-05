// standard libraries
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

// Our stuff
#include "utils.h"
#include "ndarray.h"
#include "vector.h"
#define LENGTH 4194304
#define ENABLE_PROFILING 1

int main() {
    vector a, b;
    int d = 123;
    double *data_a, *data_b;
    vector * output;
    bool flag;

    gpu_matrix_init();

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

    puts("AXPY TEST:");
    output = gpu_matrix_vector_axpy(&a, 12.0, &b);
    flag = true;
    for (index_t i = 0 ; i < LENGTH ; i++) {
        if (fabs(a.data[i]  * 12 + b.data[i] - output->data[i]) > 0.001) {
            flag = false;
            break;
        }
    }
    puts(flag ? "CORRECT!\n" : "WRONG!\n");
    puts("VECTOR_DOT TEST:");
    double dot_output = gpu_matrix_vector_dot(&a, &b);
    double expected_dot_output = 0.0;
    for (index_t i = 0;  i < LENGTH ; i++) {
        expected_dot_output += a.data[i] * b.data[i];
    }

    if (fabs(expected_dot_output - dot_output) / expected_dot_output <= 0.001) {
        puts("CORRECT!");
    } else {
        puts("WRONG!");
    }

    return 0;
}
