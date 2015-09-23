#include "kernels.h"

const char * get_program_file_name(kernel_type_t k) {
    // wtf: 
    static const char * file_names[] = {
        "resources/opencl/arimethic.cl",
        "resources/opencl/arimethic.cl",
        "resources/opencl/arimethic.cl",
        "resources/opencl/arimethic.cl",
        "resources/opencl/arimethic.cl",
        "resources/opencl/arimethic.cl",
        "resources/opencl/arimethic.cl",
        "resources/opencl/arimethic.cl",
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
    const char * file_contents = slurp(filename);
    const char * cl_function_name = get_cl_function_name(kernel_type);

    program = clCreateProgramWithSource(
        context, 1, (const char **) &file_contents,
        NULL, &status
    );
    status = clBuildProgram(
        program, 1, &device, NULL, NULL, NULL
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
