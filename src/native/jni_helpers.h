#ifndef JNI_HELPERS_H
#define JNI_HELPERS_H

#include <jni.h>
#include "types.h"

vector * retrieve_vector(JNIEnv *, jobject);
jobject package_vector(JNIEnv *, const vector *);
ndarray * retrieve_ndarray(JNIEnv *, jobject);
jobject package_ndarray(JNIEnv *, const ndarray *);

#endif
