#include <stdlib.h>
#include <stdio.h>

#include "jni_helpers.h"
#include "ndarray_jni.h"
#include "types.h"
#include "ndarray.h"
#include "utils.h"

// coerce jlong* to index_t*
// creates a new instance in memory
index_t* jlong_to_index_t(jlong* arr, index_t len) {
    index_t* ret = malloc(len * sizeof(index_t));
    for (index_t i = 0 ; i < len ; i++) {
        ret[i] = (index_t) arr[i];
    }

    return ret;
}

JNIEXPORT jobject JNICALL Java_gpu_matrix_NDArray_newScalar__D
  (JNIEnv * env, jclass klass, jdouble v) {
    ndarray * arr = ndarray_constructor_from_scalar(v);
    return package_ndarray(env, arr);
}

JNIEXPORT jobject JNICALL Java_gpu_matrix_NDArray_newScalar__
  (JNIEnv * env, jclass klass) {
    ndarray * arr = ndarray_constructor_from_scalar(0.0);
    return package_ndarray(env, arr);
}

JNIEXPORT jobject JNICALL Java_gpu_matrix_NDArray_createFromShape
  (JNIEnv * env, jclass klass, jlongArray shape) {

    jboolean is_copy;
    index_t len = (*env)->GetArrayLength(env, shape);
    jlong * long_shape = (*env)->GetLongArrayElements(env, shape, &is_copy);
    index_t * index_t_shape = jlong_to_index_t(long_shape, len);
    jobject ret = package_ndarray(
        env,
        ndarray_constructor_from_shape(len, index_t_shape)
    );

    if (is_copy) {
        free(long_shape);
    }
    free(index_t_shape);

    return ret;
}

JNIEXPORT jobject JNICALL Java_gpu_matrix_NDArray_newInstance
  (JNIEnv * env, jclass klass, jdoubleArray data,
   jlong ndims, jlongArray shape, jlongArray strides) {

    index_t * shape_index_t = jlong_to_index_t((*env)->GetLongArrayElements(env, shape, 0), ndims);
    index_t * strides_index_t = jlong_to_index_t((*env)->GetLongArrayElements(env, strides, 0), ndims);

    jobject ret = package_ndarray(
        env,
        ndarray_constructor(
            (*env)->GetDoubleArrayElements(env, data, 0),
            (index_t) ndims,
            shape_index_t,
            strides_index_t
        )
    );

    free(shape_index_t);
    free(strides_index_t);

    return ret;
}

JNIEXPORT jobject JNICALL Java_gpu_matrix_NDArray_sample
  (JNIEnv * env, jclass klass) {
    ndarray * arr = malloc(sizeof(ndarray));
    double * data = (double*) malloc(3 * sizeof(double));
    index_t * shapes = malloc(sizeof(index_t));
    index_t * strides = malloc(sizeof(index_t));
    
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

#define NDARRAY_JNI_ARIMETHIC_FACTORY(op_name) \
\
JNIEXPORT jobject JNICALL Java_gpu_matrix_NDArray_##op_name##__Lgpu_matrix_NDArray_2\
  (JNIEnv * env, jobject this, jobject other) {\
    ndarray * arr_x = retrieve_ndarray(env, this);\
    ndarray * arr_y = retrieve_ndarray(env, other);\
    return package_ndarray(env, ndarray_##op_name(arr_x, arr_y));\
}\
\
JNIEXPORT jobject JNICALL Java_gpu_matrix_NDArray_##op_name##__D \
  (JNIEnv * env, jobject this, jdouble y) {\
    ndarray * arr = retrieve_ndarray(env, this);\
    return package_ndarray(env, ndarray_##op_name##_scalar(arr, (double) y));\
}\
\
JNIEXPORT jobject JNICALL Java_gpu_matrix_NDArray_##op_name##__Lgpu_matrix_Vector_2 \
  (JNIEnv * env, jobject this, jobject other) {\
    ndarray * arr_x = retrieve_ndarray(env, this); \
    vector * v_y = retrieve_vector(env, other);\
    return package_ndarray(env, ndarray_##op_name##_vector(arr_x, v_y));\
} \

NDARRAY_JNI_ARIMETHIC_FACTORY(add);
NDARRAY_JNI_ARIMETHIC_FACTORY(sub);
NDARRAY_JNI_ARIMETHIC_FACTORY(mul);
NDARRAY_JNI_ARIMETHIC_FACTORY(div);

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
    const jlong * indexes = (*env)->GetLongArrayElements(env, indexes_arg, 0);
    index_t pos = 0;

    for(int i = 0 ; i < ndims ; i++) {
        pos += ((index_t) indexes[i]) * strides[i];
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
  (JNIEnv * env, jobject this, jlongArray indexes_arg, jdouble v) {
    ndarray * arr = retrieve_ndarray(env, this);
    index_t * indexes = jlong_to_index_t(
        (*env)->GetLongArrayElements(env, indexes_arg, 0),
        arr->ndims);
    ndarray_set_nd(
        arr, indexes, v
    );
    return this;
}

JNIEXPORT jobject JNICALL Java_gpu_matrix_NDArray_fill
  (JNIEnv * env, jobject this, jdouble v) {
    ndarray * arr = retrieve_ndarray(env, this);
    for (index_t i = 0 ; i < ndarray_elements_count(arr) ; i++) {
        arr->data[i] = (double) v;
    }
    return this;
}

JNIEXPORT jobject JNICALL Java_gpu_matrix_NDArray_assign__D
  (JNIEnv * env, jobject this, jdouble v) {
    ndarray * arr = retrieve_ndarray(env, this);
    for (index_t i = 0 ; i < ndarray_elements_count(arr) ; i++) {
        arr->data[i] = (double) v;
    }
    return this;
}

JNIEXPORT jobject JNICALL Java_gpu_matrix_NDArray_assign__Lgpu_matrix_NDArray_2
  (JNIEnv * env, jobject this, jobject obj) {
    ndarray * arr = retrieve_ndarray(env, this); 
    ndarray * src = ndarray_broadcast(
        retrieve_ndarray(env, obj),
        arr->ndims,
        arr->shape
    );

    // Free the value holders of arr stuff
    free(arr->data);
    free(arr->shape);
    free(arr->strides);

    // free the src 
    free(src);

    arr->data = src->data;
    arr->shape = src->shape;
    arr->strides = src->strides;

     
    return this;
}

JNIEXPORT jboolean JNICALL Java_gpu_matrix_NDArray_equals__Lgpu_matrix_NDArray_2
  (JNIEnv * env, jobject this, jobject obj) {
    ndarray * arr_x = retrieve_ndarray(env, this);
    ndarray * arr_y = retrieve_ndarray(env, obj);

    return (arr_x == arr_y ||
        ndarray_equals(arr_x, arr_y));
}

JNIEXPORT jboolean JNICALL Java_gpu_matrix_NDArray_equals__Lgpu_matrix_NDArray_2D
  (JNIEnv * env, jobject this, jobject obj, jdouble eps) {
    ndarray * arr_x = retrieve_ndarray(env, this);
    ndarray * arr_y = retrieve_ndarray(env, obj);
    
    return (arr_x == arr_y ||
        ndarray_equals_epsilon(arr_x, arr_y, eps));
}

JNIEXPORT jboolean JNICALL Java_gpu_matrix_NDArray_equals__D
  (JNIEnv * env, jobject this, jdouble y) {

    ndarray * arr = retrieve_ndarray(env, this);
    return ndarray_equals_scalar(arr, (double) y);
}

JNIEXPORT jdoubleArray JNICALL Java_gpu_matrix_NDArray_flatten
  (JNIEnv * env, jobject this) {
    ndarray * arr = retrieve_ndarray(env, this);
    index_t count = ndarray_elements_count(arr);
    double * data = ndarray_flatten(arr);
    jdoubleArray ret = (*env)->NewDoubleArray(
        env, count
    );
    (*env)->SetDoubleArrayRegion(
        env, ret, 0, count, data
    );
    free(data);
    return ret;
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
        printf("%lu ", (unsigned long) shape[i]);
    }
    puts("");

    // print strides
    puts("Stides:");
    for(int i = 0 ; i < ndims ; i++) {
        printf("%lu ", (unsigned long) strides[i]);
    }
    puts("");
    
    puts("------");
    fflush(stdout);
}

JNIEXPORT jobject JNICALL Java_gpu_matrix_NDArray_mmul
  (JNIEnv * env, jobject this, jobject obj) {
    ndarray * arr_x = retrieve_ndarray(env, this);
    ndarray * arr_y = retrieve_ndarray(env, obj);

    return package_ndarray(
        env,
        ndarray_mmul(arr_x, arr_y)
    );
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
