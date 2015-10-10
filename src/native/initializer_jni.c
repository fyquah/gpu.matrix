#include "initializer_jni.h"
#include "types.h"
#include "utils/kernels.h"
#include "utils/cl_helper.h"

JNIEXPORT void JNICALL Java_gpu_matrix_Initializer_init
  (JNIEnv * env, jclass klass) {
    gpu_matrix_init();
    // gpu_matrix_kernel_set_jvm(env);
}

