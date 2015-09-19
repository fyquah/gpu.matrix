#include <CL/cl.h>
#include <stdlib.h>
#include <stdio.h>

#include "ndarray_jni.h"
#include "ndarray.h"
#include "utils.h"
   
// Utils
ndarray * retrieve_ndarray(JNIEnv * env, jobject this) {

    jclass cls = (*env)->GetObjectClass(env, this);
    jfieldID fid = (*env)->GetFieldID(env, cls, "bb", "Ljava/nio/ByteBuffer;");
    jobject bb = (*env)->GetObjectField(env, this, fid);
    ndarray * data = (ndarray*) (*env)->GetDirectBufferAddress(env, bb);
    return data;
}

jobject package_ndarray(JNIEnv * env, const ndarray * data) {
    jclass klass = (*env)->FindClass(env, "gpu/matrix/NDArray");
    jmethodID constructor = (*env)->GetMethodID(env, klass, "<init>", "()V");

    // instantiate the object
    jobject obj = (*env)->NewObject(env, klass, constructor);   
    jfieldID fid = (*env)->GetFieldID(env, klass, "bb", "Ljava/nio/ByteBuffer;");

    // set the ndarray pointer to the bb field of the object
    jobject bb = (*env)->NewDirectByteBuffer(env, (void*) data, sizeof(ndarray));
    (*env)->SetObjectField(env, obj, fid, bb);
    
    return obj;
}

// static methods

JNIEXPORT void JNICALL Java_gpu_matrix_NDArray_init
  (JNIEnv * env, jclass klass) {
    gpu_matrix_init();
}

JNIEXPORT jobject JNICALL Java_gpu_matrix_NDArray_newInstance
  (JNIEnv * env, jclass klass, jdoubleArray data,
   jlong ndims, jlongArray shape, jlongArray strides) {

    return package_ndarray(
        env,
        ndarray_constructor(
            (*env)->GetDoubleArrayElements(env, data, 0),
            (long) ndims,
            (*env)->GetLongArrayElements(env, shape, 0),
            (*env)->GetLongArrayElements(env, strides, 0)
        )
    );
}

JNIEXPORT jobject JNICALL Java_gpu_matrix_NDArray_sample
  (JNIEnv * env, jclass klass) {
    ndarray * arr = malloc(sizeof(ndarray));
    double * data = (double*) malloc(3 * sizeof(double));
    size_t * shapes = malloc(sizeof(unsigned));
    size_t * strides = malloc(sizeof(unsigned));
    
    data[0] = 6.0;
    data[1] = 5.0;
    data[2] = 7.0;
    shapes[0] = 3;
    strides[0] = 1;

    arr->data = data;
    arr->shape = shapes;
    arr->strides = strides;
    arr->ndims = 1;

    return package_ndarray(env, arr);
}

// object methods

JNIEXPORT jobject JNICALL Java_gpu_matrix_NDArray_add
  (JNIEnv * env, jobject this, jobject other) {
    ndarray * arr_x = retrieve_ndarray(env, this);
    ndarray * arr_y = retrieve_ndarray(env, other);
    return package_ndarray(env, ndarray_add(arr_x, arr_y));
}

JNIEXPORT void JNICALL Java_gpu_matrix_NDArray_print
  (JNIEnv * env, jobject this) {
    ndarray * arr = retrieve_ndarray(env, this);
    double * data = arr->data;
    size_t * strides = arr->strides;
    size_t * shape = arr->shape;
    unsigned ndims = arr->ndims;

    // dump data
    puts("Data:");
    for(int i = 0 ; i < ndarray_elements_count(arr) ; i++) {
        printf("%.f ", data[i]);
    }
    puts("");

    // print shape
    puts("Shape:");
    for(int i = 0 ; i < ndims ; i++) {
        printf("%lu ", shape[i]);
    }
    puts("");

    // print strides
    puts("Stides:");
    for(int i = 0 ; i < ndims ; i++) {
        printf("%lu ", strides[i]);
    }
    puts("");
    
    puts("------");
    fflush(stdout);
}

