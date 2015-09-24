#include "jvmloader_jni.h"
#include "types.h"
#include "utils/kernels.h"

JNIEXPORT void JNICALL Java_gpu_matrix_JVMLoader_init
  (JNIEnv * env, jclass klass) {
    gpu_matrix_kernel_set_jvm(env);
}
