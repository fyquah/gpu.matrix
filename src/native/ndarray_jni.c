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

JNIEXPORT jobject JNICALL Java_gpu_matrix_NDArray_createFromShape
  (JNIEnv * env, jclass klass, jlongArray shape) {
    return package_ndarray(
        env,
        ndarray_constructor_from_shape(
            (index_t) (*env)->GetArrayLength(env, shape),
            (index_t*) (*env)->GetLongArrayElements(env, shape, 0)
        )
    );
}

JNIEXPORT jobject JNICALL Java_gpu_matrix_NDArray_newInstance
  (JNIEnv * env, jclass klass, jdoubleArray data,
   jlong ndims, jlongArray shape, jlongArray strides) {

    return package_ndarray(
        env,
        ndarray_constructor(
            (*env)->GetDoubleArrayElements(env, data, 0),
            (index_t) ndims,
            (index_t*) (*env)->GetLongArrayElements(env, shape, 0),
            (index_t*) (*env)->GetLongArrayElements(env, strides, 0)
        )
    );
}

JNIEXPORT jobject JNICALL Java_gpu_matrix_NDArray_sample
  (JNIEnv * env, jclass klass) {
    ndarray * arr = malloc(sizeof(ndarray));
    double * data = (double*) malloc(3 * sizeof(double));
    index_t * shapes = malloc(sizeof(unsigned));
    index_t * strides = malloc(sizeof(unsigned));
    
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

JNIEXPORT jobject JNICALL Java_gpu_matrix_NDArray_clone
  (JNIEnv * env, jobject this) {
    return package_ndarray(
        env,
        ndarray_clone(
            retrieve_ndarray(env, this)
        )
    );
}

JNIEXPORT jobject JNICALL Java_gpu_matrix_NDArray_add
  (JNIEnv * env, jobject this, jobject other) {
    ndarray * arr_x = retrieve_ndarray(env, this);
    ndarray * arr_y = retrieve_ndarray(env, other);
    return package_ndarray(env, ndarray_add(arr_x, arr_y));
}

JNIEXPORT jdouble JNICALL Java_gpu_matrix_NDArray_get__J
  (JNIEnv * env, jobject this, jlong i) {
    ndarray * arr = retrieve_ndarray(env, this);
    // obviously, the strides is 1 for a vector
    return arr->data[i];
}

JNIEXPORT jdouble JNICALL Java_gpu_matrix_NDArray_get__JJ
  (JNIEnv * env, jobject this, jlong i, jlong j) {
    ndarray * arr = retrieve_ndarray(env, this);
    index_t * strides = arr->strides;
    return arr->data[i * strides[0] + j * strides[1]];
}

JNIEXPORT jdouble JNICALL Java_gpu_matrix_NDArray_get___3J
  (JNIEnv * env, jobject this, jlongArray indexes_arg) {
    ndarray * arr = retrieve_ndarray(env, this);
    const index_t ndims = arr->ndims;
    const index_t * strides = arr->strides;
    const index_t * indexes = (index_t*) (*env)->GetLongArrayElements(env, indexes_arg, 0);
    index_t pos = 0;

    for(int i = 0 ; i < ndims ; i++) {
        pos += indexes[i] * strides[i];
    }

    return arr->data[pos];
}

JNIEXPORT jobject JNICALL Java_gpu_matrix_NDArray_set__JD
  (JNIEnv * env, jobject this, jlong i, jdouble v) {
    ndarray * arr = retrieve_ndarray(env, this);
    ndarray_set_1d(arr, i, v);
    return this;
}

JNIEXPORT jobject JNICALL Java_gpu_matrix_NDArray_set__JJD
  (JNIEnv * env, jobject this, jlong i, jlong j, jdouble v) {
    ndarray * arr = retrieve_ndarray(env, this);
    ndarray_set_2d(arr, i, j, v);
    return this;
}

JNIEXPORT jobject JNICALL Java_gpu_matrix_NDArray_set___3JD
  (JNIEnv * env, jobject this, jlongArray indexes, jdouble v) {
    ndarray * arr = retrieve_ndarray(env, this);
    ndarray_set_nd(
        arr,
        (*env)->GetLongArrayElements(env, indexes, 0),
        v
    );
    return this;
}

JNIEXPORT void JNICALL Java_gpu_matrix_NDArray_print
  (JNIEnv * env, jobject this) {
    ndarray * arr = retrieve_ndarray(env, this);
    double * data = arr->data;
    index_t * strides = arr->strides;
    index_t * shape = arr->shape;
    unsigned ndims = arr->ndims;

    // dump data
    puts("Data:");
    for(int i = 0 ; i < ndarray_elements_count(arr) ; i++) {
        printf("%.5f ", data[i]);
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

JNIEXPORT void JNICALL Java_gpu_matrix_NDArray_finalize
  (JNIEnv * env, jobject this) {
    ndarray_release(retrieve_ndarray(env, this));

    jclass klass = (*env)->GetObjectClass(env, this);
    jclass super_klass = (*env)->GetSuperclass(env, klass);
    jmethodID method = (*env)->GetMethodID(env, super_klass, "finalize", "()V");

    // Call super class method...
    (*env)->CallNonvirtualVoidMethod(env, this, super_klass, method);
}

JNIEXPORT jlong JNICALL Java_gpu_matrix_NDArray_dimensionality
  (JNIEnv * env, jobject this){
    return (jlong) (retrieve_ndarray(env, this))->ndims;       
}

JNIEXPORT jlongArray JNICALL Java_gpu_matrix_NDArray_shape
  (JNIEnv * env, jobject this) {
    ndarray * arr = retrieve_ndarray(env, this);
    index_t ndims = arr->ndims;
    jlongArray results = (*env)->NewLongArray(env, arr->ndims);
    jlong * fill = malloc(sizeof(jlong) * ndims);
    for(int i = 0 ; i < ndims; i++) {
        fill[i] = arr->shape[i];
    }
    (*env)->SetLongArrayRegion(env, results, 0, ndims, fill);
    free(fill);

    return results;
}
