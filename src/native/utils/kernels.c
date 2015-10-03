#include "kernels.h"

#define RESOURCES_PREFIX "../../resources/"

// note: TRY_AND_CATCH_ERROR macro is defined everytime it is required - 
// we may require a slightly different version for depending on where it is used
// (eg: sometimes we want to return, sometimes we just want to exit)
// behavious is largely similiar in the same file, but should vary in different ones
#define TRY_AND_CATCH_ERROR(statement, status_var) \
statement;\
if (status_var != CL_SUCCESS) { \
    fprintf(stderr, "An error occured in running gpu.matrix at line %u of %s\n" , \
            __LINE__, __FILE__);  \
    exit(1); \
}

// global buffers
static JavaVM * jvm;                     // to load resources' path
cl_kernel kernels_buffer[KERNELS_COUNT]; // to cache kernels
cl_program program_buffer;               // to cache programs

void gpu_matrix_kernel_set_jvm(JNIEnv * env) {
    jint rs = (*env)->GetJavaVM(env, &jvm);
    assert(rs == JNI_OK);
}

static char * get_file_contents(const char * filename) {
    char full_path[10000];

    if (jvm == NULL) {
        // JVM is not initialized, load from resources directory 
        strcpy(full_path, RESOURCES_PREFIX);
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
        klass = (*env)->FindClass(env, "gpu/matrix/LoaderUtils");
        method = (*env)->GetStaticMethodID(env,
            klass, "clLoadProgram",
            "(Ljava/lang/String;)Ljava/lang/String;");
        arg = (jobject) (*env)->NewStringUTF(env, full_path);
        result = (*env)->CallStaticObjectMethod(env, klass, method, arg);

        // load results into a C string pointer
        contents_length = (*env)->GetStringUTFLength(env, (jstring) result);
        contents_holder = (char*) (*env)->GetStringUTFChars(env, (jstring) result, NULL);
        contents = malloc((1 + contents_length) * sizeof(char));
        for (int i = 0 ; i < contents_length ; i++) {
            contents[i] = contents_holder[i];
        }
        contents[contents_length] = '\0';
        (*env)->ReleaseStringUTFChars(env, (jstring) result, contents_holder);

        return contents;
    }
}

const char * get_cl_function_name(kernel_type_t k) {
    static const char * module_names[] = {
        "add", "add_scalar",
        "sub", "sub_scalar",
        "mul", "mul_scalar",
        "div", "div_scalar",
        "mmul", "vector_axpy"
    };

    return module_names[k];
}

char * get_compilation_options() {
    if (jvm == NULL) {
        char * results = malloc(1 + strlen("-I " RESOURCES_PREFIX "opencl/"));
        strcpy(results, "-I " RESOURCES_PREFIX "opencl/");
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
        klass = (*env)->FindClass(env, "gpu/matrix/LoaderUtils");
        method = (*env)->GetStaticMethodID(env,
            klass, "clGetCompilationFlags",
            "()Ljava/lang/String;");
        result = (*env)->CallStaticObjectMethod(env, klass, method);

        // load results into a C string pointer
        contents_length = (*env)->GetStringUTFLength(env, (jstring) result);
        contents_holder = (char*) (*env)->GetStringUTFChars(
                env, (jstring) result, NULL);
        contents = malloc((1 + contents_length) * sizeof(char));
        for (int i = 0 ; i < contents_length ; i++) {
            contents[i] = contents_holder[i];
        }
        contents[contents_length] = '\0';
        (*env)->ReleaseStringUTFChars(env, (jstring) result, contents_holder);

        return contents;
    }
}

static cl_program compile_program(
        cl_context context,
        cl_device_id device
    ) {
    cl_int status;
    cl_program program;
    char * file_contents, *build_options;

    // loading the file contents
    file_contents = get_file_contents("main.cl");

    // create the program
    // TODO: Load program from binary if it exists
    // Overhead here is acceptable, as this function is called only once
    TRY_AND_CATCH_ERROR(
        program = clCreateProgramWithSource(
            context, 1, (const char**) &file_contents,
            NULL, &status
        );,
        status
    );

    // compile the program
    build_options = get_compilation_options();
    status = clBuildProgram(
        program, 1, &device,
        (const char*) build_options,
        NULL, NULL
    );

    // catch any compilation errors here
    if (status != CL_SUCCESS) {
        if (status == CL_BUILD_PROGRAM_FAILURE) {
            size_t log_size;
            char * log;
            clGetProgramBuildInfo(
                program,
                device,
                CL_PROGRAM_BUILD_LOG,
                0,
                NULL,
                &log_size
            );

            // Allocate memory for the log
            log = (char *) malloc(log_size);

            // Get the log
            clGetProgramBuildInfo(
                program,
                device,
                CL_PROGRAM_BUILD_LOG,
                log_size,
                log,
                NULL
            );

            // Print the log
            fprintf(stderr, "%s\n", log);

            // exit program!
        }
        fprintf(stderr, "An error occured at line %u in the file %s",
            __LINE__, __FILE__);
        exit(1);
    }

    free((void*) build_options);
    free((void*) file_contents);

    // cache the results
    program_buffer = program;

    return program;
}

cl_program program_get(cl_context context, cl_device_id device) {
    if (program_buffer == NULL) {
        return compile_program(context, device);
    } else {
        return program_buffer;
    }
}

static cl_kernel compile_kernel(
        cl_context context,
        cl_device_id device,
        kernel_type_t kernel_type
    ) {
    cl_int status;
    cl_kernel kernel;
    cl_program program;
    const char * cl_function_name = get_cl_function_name(kernel_type);
 
    // Try obtaining the cached program or compiling the 
    // program from scartch
    program = program_get(context, device);

    TRY_AND_CATCH_ERROR(
        kernel = clCreateKernel(
            program,
            cl_function_name,
            &status
        );,
        status
    );

    // cache the results
    kernels_buffer[kernel_type] = kernel; 

    return kernel;
}

cl_kernel kernels_get(cl_context context, cl_device_id device, kernel_type_t kernel_type) {

    if (kernels_buffer[kernel_type] != NULL) {
        return kernels_buffer[kernel_type];
    } else {
        return compile_kernel(context, device, kernel_type); 
    }
}

