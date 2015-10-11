#include "jni_helpers.h"

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
