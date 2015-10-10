#include "vector_jni.h"
#include "vector.h"
#include "types.h"
/* Header for class gpu_matrix_Vector */

vector * retrieve_vector(JNIEnv * env, jobject this) {
    jclass cls = (*env)->GetObjectClass(env, this);
    jfieldID fid = (*env)->GetFieldID(env, cls, "bb", "Ljava/nio/ByteBuffer;");
    jobject bb = (*env)->GetObjectField(env, this, fid);
    vector * v = (vector*) (*env)->GetDirectBufferAddress(env, bb);
    return v;
}

jobject package_vector(JNIEnv * env, const vector * data) {

    jclass klass = (*env)->FindClass(env, "gpu/matrix/Vector");
    jmethodID constructor = (*env)->GetMethodID(env, klass, "<init>", "()V");

    // instantiate the object
    jobject obj = (*env)->NewObject(env, klass, constructor);   
    jfieldID fid = (*env)->GetFieldID(env, klass, "bb", "Ljava/nio/ByteBuffer;");

    // set the ndarray pointer to the bb field of the object
    jobject bb = (*env)->NewDirectByteBuffer(env, (void*) data, sizeof(vector));
    (*env)->SetObjectField(env, obj, fid, bb);
    
    return obj;
}


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

JNIEXPORT void JNICALL Java_gpu_matrix_Vector_print
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
}

JNIEXPORT jobject JNICALL Java_gpu_matrix_Vector_newInstance
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

JNIEXPORT void JNICALL Java_gpu_matrix_Vector_finalize
  (JNIEnv * env, jobject this) {
    // TODO : Free memory

    jclass klass = (*env)->GetObjectClass(env, this);
    jclass super_klass = (*env)->GetSuperclass(env, klass);
    jmethodID method = (*env)->GetMethodID(env, super_klass, "finalize", "()V");

    // Call super class method...
    (*env)->CallNonvirtualVoidMethod(env, this, super_klass, method);
}

