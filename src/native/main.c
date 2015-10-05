// standard libraries
#include <assert.h>
#include "tests/test_vector.h"
#include "utils.h"

int main() {
    gpu_matrix_init();
    test_vector();
    return 0;
}
