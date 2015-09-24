#include "kernels.h"

#define SOURCE_PREFIX "../../resources/"

static JavaVM * jvm;

void gpu_matrix_kernel_set_jvm(JNIEnv * env) {
    jint rs = (*env)->GetJavaVM(env, &jvm);
    assert(rs == JNI_OK);
}

char * get_file_contents(const char * filename) {
    char full_path[10000];

    if (jvm == NULL) {
        // JVM is not initialized, load from resources directory 
        strcpy(full_path, "resources/");
        strcat(full_path, "opencl/");
        strcat(full_path, filename);
        char * contents = slurp(full_path);
        return contents;

    } else {
        // JVM is initialized, load from JVM
        JNIEnv * env;
        jint rs;
        jclass klass;
        jmethodID method;
        jstring arg, result;
        unsigned contents_length;
        char * contents_holder;
        char * contents;

        // load the full_path
        strcpy(full_path, filename);

        rs = (*jvm)->AttachCurrentThread(jvm, (void**) &env, NULL);
        assert(rs == JNI_OK);
        klass = (*env)->FindClass(env, "gpu/matrix/KernelLoader");
        method = (*env)->GetStaticMethodID(env,
            klass, "loadProgram",
            "(Ljava/lang/String;)Ljava/lang/String;");
        arg = (jobject) (*env)->NewStringUTF(env, full_path);
        result = (*env)->CallStaticObjectMethod(env, klass, method, arg);

        // load results into a C string pointer
        contents_length = (*env)->GetStringUTFLength(env, (jstring) result);
        contents_holder = (*env)->GetStringUTFChars(env, (jstring) result, NULL);
        contents = malloc((1 + contents_length) * sizeof(char));
        for (int i = 0 ; i < contents_length ; i++) {
            contents[i] = contents_holder[i];
        }
        contents[contents_length] = '\0';
        (*env)->ReleaseStringUTFChars(env, (jstring) result, contents_holder);

        return contents;
    }
}

const char * get_program_file_name(kernel_type_t k) {
    // wtf: 
    static const char * file_names[] = {
        "arimethic.cl",
        "arimethic.cl",
        "arimethic.cl",
        "arimethic.cl",
        "arimethic.cl",
        "arimethic.cl",
        "arimethic.cl",
        "arimethic.cl",
    };

    return file_names[k];
}

const char * get_cl_function_name(kernel_type_t k) {
    static const char * module_names[] = {
        "add", "add_scalar",
        "sub", "sub_scalar",
        "mul", "mul_scalar",
        "div", "div_scalar"
    };

    return module_names[k];
}

const char * get_source_include_directory() {
    if (jvm == NULL) {
        char * results = malloc(1 + strlen("-I " SOURCE_PREFIX "opencl"));
        strcpy(results, "-I " SOURCE_PREFIX "opencl");
        return results;
    } else {
        jint rs;
        jobject result;
        unsigned contents_length;
        char * contents_holder;
        char * contents;
        JNIEnv * env;
        jmethodID method;
        jclass klass;

        rs = (*jvm)->AttachCurrentThread(jvm, (void**) &env, NULL);
        assert(rs == JNI_OK);
        klass = (*env)->FindClass(env, "gpu/matrix/KernelLoader");
        method = (*env)->GetStaticMethodID(env,
            klass, "getIncludeProgramDir",
            "()Ljava/lang/String;");
        result = (*env)->CallStaticObjectMethod(env, klass, method);

        // load results into a C string pointer
        contents_length = (*env)->GetStringUTFLength(env, (jstring) result);
        contents_holder = (*env)->GetStringUTFChars(env, (jstring) result, NULL);
        contents = malloc((1 + contents_length) * sizeof(char));
        for (int i = 0 ; i < contents_length ; i++) {
            contents[i] = contents_holder[i];
        }
        contents[contents_length] = '\0';
        (*env)->ReleaseStringUTFChars(env, (jstring) result, contents_holder);

        return contents;
    }
}

cl_kernel kernels_get(cl_context context, cl_device_id device, kernel_type_t kernel_type) {
    cl_int status;
    cl_kernel kernel;
    cl_program program;
    char * include_dir;
    const char * filename = get_program_file_name(kernel_type);
    const char * file_contents = get_file_contents(filename);
    const char * cl_function_name = get_cl_function_name(kernel_type);

    program = clCreateProgramWithSource(
        context, 1, (const char **) &file_contents,
        NULL, &status
    );
    include_dir = get_source_include_directory();
    status = clBuildProgram(
        program, 1, &device,
        include_dir,
        NULL, NULL
    );
    free(include_dir);
    if (status == CL_BUILD_PROGRAM_FAILURE) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

        // Allocate memory for the log
        char *log = (char *) malloc(log_size);

        // Get the log
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

        // Print the log
        printf("%s\n", log);
    }
    kernel = clCreateKernel(program, cl_function_name, &status);
    free((void*) file_contents);
    clReleaseProgram(program);
    return kernel;
}
