#include "jni_helpers.h"
#include "vector_jni.h"
#include "jni_helpers.h"
#include "ndarray.h"
#include "vector.h"
#include "types.h"
/* Header for class gpu_matrix_Vector */

JNIEXPORT jobject JNICALL Java_gpu_matrix_Vector_axpy
  (JNIEnv * env, jobject this, jdouble alpha, jobject other) {
    vector * v_x = retrieve_vector(env, this);
    vector * v_y = retrieve_vector(env, other);
    gpu_matrix_vector_axpy(v_x, alpha, v_y);
    
    return this;
}

/*
 * Class:     gpu_matrix_Vector
 * Method:    scal
 * Signature: (D)Lgpu/matrix/Vector;
 */
JNIEXPORT jobject JNICALL Java_gpu_matrix_Vector_scal
  (JNIEnv * env, jobject this, jdouble alpha) {
    vector * v_x = retrieve_vector(env, this);
    gpu_matrix_vector_scal(v_x, (double) alpha);
    return this;
}

/*
 * Class:     gpu_matrix_Vector
 * Method:    clone
 * Signature: ()Lgpu/matrix/Vector;
 */
JNIEXPORT jobject JNICALL Java_gpu_matrix_Vector_clone
  (JNIEnv * env, jobject this) {
    vector * v_x = retrieve_vector(env, this);
    return package_vector(
        env,
        gpu_matrix_vector_copy(v_x)
    );
}

/*
 * Class:     gpu_matrix_Vector
 * Method:    dot
 * Signature: (Lgpu/matrix/Vector;)D
 */
JNIEXPORT jdouble JNICALL Java_gpu_matrix_Vector_dot
  (JNIEnv * env, jobject this, jobject other) {
    vector * v_x = retrieve_vector(env, this);
    vector * v_y = retrieve_vector(env, other);
    return gpu_matrix_vector_dot(v_x, v_y);
}

/*
 * Class:     gpu_matrix_Vector
 * Method:    nrm2
 * Signature: ()D
 */
JNIEXPORT jdouble JNICALL Java_gpu_matrix_Vector_nrm2
  (JNIEnv * env, jobject this) {
    vector * v_x = retrieve_vector(env, this);
    return gpu_matrix_vector_nrm2(v_x);
  
}

/*
 * Class:     gpu_matrix_Vector
 * Method:    asum
 * Signature: ()D
 */
JNIEXPORT jdouble JNICALL Java_gpu_matrix_Vector_asum
  (JNIEnv * env, jobject this) {
    vector * v_x = retrieve_vector(env, this);
    return gpu_matrix_vector_asum(v_x);
}

JNIEXPORT jobject JNICALL Java_gpu_matrix_Vector_print
  (JNIEnv * env, jobject this) {
    vector * v = retrieve_vector(env, this);
    printf("Length: %u, Strides: %u\n", v->length, v->stride);
    printf("[");
    for (index_t i = 0 ; i < v->length ; i++) {
        if (i != 0) {
            printf(", ");
        }
        printf("%.3f", v->data[i]);
    }
    printf("]\n");
    
    fflush(stdout);
    return this;
}

JNIEXPORT jdoubleArray JNICALL Java_gpu_matrix_Vector_toArray
  (JNIEnv * env, jobject this) {
    vector * v = retrieve_vector(env, this);
    jdoubleArray ret = (*env)->NewDoubleArray(env, v->length);
    if(v->stride == 1) {
        (*env)->SetDoubleArrayRegion(env, ret, 0, v->length, v->data);
    } else {
        // make a duplicate, exploiting cacche locality
        double * tmp = calloc(v->length, sizeof(double));
        for (index_t i = 0 ; i < v->length; i++) {
            tmp[i] = v->data[i * v->stride];
        }
        (*env)->SetDoubleArrayRegion(env, ret, 0, v->length, v->data);
        free(tmp);
    }

    return ret;
}

#define GPU_MATRIX_VECTOR_JNI_ARIMETHIC_FACTORY(name) \
JNIEXPORT jobject JNICALL Java_gpu_matrix_Vector_##name##__D \
  (JNIEnv * env, jobject this, jdouble alpha) { \
    vector * v_x = retrieve_vector(env, this); \
    vector * ret = gpu_matrix_vector_##name##_scalar(v_x, (double) alpha); \
    return package_vector(env, ret); \
} \
\
JNIEXPORT jobject JNICALL Java_gpu_matrix_Vector_##name##__Lgpu_matrix_Vector_2 \
  (JNIEnv * env, jobject this, jobject other) { \
    vector * v_x = retrieve_vector(env, this); \
    vector * v_y = retrieve_vector(env, other); \
    vector * ret = gpu_matrix_vector_##name##_2(v_x, v_y); \
    return package_vector(env, ret); \
} \
JNIEXPORT jobject JNICALL Java_gpu_matrix_Vector_##name##__Lgpu_matrix_NDArray_2 \
  (JNIEnv * env, jobject this, jobject other) { \
    vector * v_x = retrieve_vector(env, this); \
    ndarray * arr_y = retrieve_ndarray(env, other); \
    ndarray * ret = ndarray_##name##_vector(arr_y, v_x); \
    return package_ndarray(env, ret); \
}

GPU_MATRIX_VECTOR_JNI_ARIMETHIC_FACTORY(add)
GPU_MATRIX_VECTOR_JNI_ARIMETHIC_FACTORY(mul)
GPU_MATRIX_VECTOR_JNI_ARIMETHIC_FACTORY(sub)
GPU_MATRIX_VECTOR_JNI_ARIMETHIC_FACTORY(div)


JNIEXPORT jobject JNICALL Java_gpu_matrix_Vector_pow
  (JNIEnv * env, jobject this, jdouble exponent) {
    vector * v = retrieve_vector(env, this);
    return package_vector(
        env,
        gpu_matrix_vector_pow(v, (double) exponent)
    );
}

JNIEXPORT jdouble JNICALL Java_gpu_matrix_Vector_sum
  (JNIEnv * env, jobject this) {
    vector * v = retrieve_vector(env, this);
    return ((double) gpu_matrix_vector_sum(v));
}

JNIEXPORT jlong JNICALL Java_gpu_matrix_Vector_length
  (JNIEnv * env, jobject this) {
    vector * v = retrieve_vector(env, this);
    jlong ret = (jlong) v->length;
    return ret;
}

JNIEXPORT jdouble JNICALL Java_gpu_matrix_Vector_get
  (JNIEnv * env, jobject this, jlong idx) {
    vector * v = retrieve_vector(env, this);
    return ((jdouble) (v->data[v->stride * ((index_t) idx)]));
}

JNIEXPORT void JNICALL Java_gpu_matrix_Vector_set
  (JNIEnv * env, jobject this, jlong idx, jdouble val) {
    vector * v = retrieve_vector(env, this);
    double coerced_val = (double) val;
    index_t coerced_idx = (index_t) idx;
    v->data[v->stride * coerced_idx] = coerced_val;
}

JNIEXPORT jobject JNICALL Java_gpu_matrix_Vector_newInstance___3D
  (JNIEnv * env, jclass klass, jdoubleArray arr) {
    vector * v;
    jboolean is_copy;
    index_t len = (*env)->GetArrayLength(env, arr);
    jdouble * data = (*env)->GetDoubleArrayElements(env, arr, &is_copy);
    v = malloc(sizeof(vector));
    v->length = len;
    v->stride = 1;
    v->data = malloc(sizeof(double) * len);

    for (index_t i = 0 ; i < len ; i++) {
        v->data[i] = (double) data[i]; 
    }

    if (is_copy) {
        (*env)->ReleaseDoubleArrayElements(env, arr, data, 0);
    }
    return package_vector(env, v);
}

JNIEXPORT jobject JNICALL Java_gpu_matrix_Vector_newInstance__J
  (JNIEnv * env, jclass klass, jlong len) {
    vector * v;
    v = malloc(sizeof(vector));
    v->length = (index_t) len;
    v->stride = 1;
    v->data = malloc(sizeof(double) * len);
    for (index_t i = 0 ; i < len ; i++) {
        v->data[i] = 0.0;
    }
    return package_vector(env, v);
}


JNIEXPORT jobject JNICALL Java_gpu_matrix_Vector_newInstance__JD
  (JNIEnv * env, jclass klass, jlong len, jdouble initial_value) {
    vector * v;
    v = malloc(sizeof(vector));
    v->length = (index_t) len;
    v->stride = 1;
    v->data = malloc(sizeof(double) * len);
    for (index_t i = 0 ; i < len ; i++) {
        v->data[i] = (double) initial_value;
    }
    return package_vector(env, v);
}

JNIEXPORT void JNICALL Java_gpu_matrix_Vector_finalize
  (JNIEnv * env, jobject this) {
    // TODO : Free memory

    jclass klass = (*env)->GetObjectClass(env, this);
    jclass super_klass = (*env)->GetSuperclass(env, klass);
    jmethodID method = (*env)->GetMethodID(env, super_klass, "finalize", "()V");

    // Call super class method...
    (*env)->CallNonvirtualVoidMethod(env, this, super_klass, method);
}

