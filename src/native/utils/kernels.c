#include "kernels.h"

#define SOURCE_PREFIX "resources/opencl/"

JNIEnv * JNI_ENV = NULL;

char * get_file_contents(const char * filename) {
    if (JNI_ENV == NULL) {
        char * full_path = malloc((strlen(SOURCE_PREFIX) + strlen(filename)) * sizeof(char));
        char * contents = slurp(full_path);
        free(full_path);
        return contents;
    } else {
        puts("Not implemented! get_file_contentx, utils/kernels.c");
        exit(1);
        return "";
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

cl_kernel kernels_get(cl_context context, cl_device_id device, kernel_type_t kernel_type) {
    cl_int status;
    cl_kernel kernel;
    cl_program program;
    const char * filename = get_program_file_name(kernel_type);
    const char * file_contents = get_file_contents(filename);
    const char * cl_function_name = get_cl_function_name(kernel_type);

    program = clCreateProgramWithSource(
        context, 1, (const char **) &file_contents,
        NULL, &status
    );
    status = clBuildProgram(
        program, 1, &device,
        "-I " SOURCE_PREFIX,
        NULL, NULL
    );
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
