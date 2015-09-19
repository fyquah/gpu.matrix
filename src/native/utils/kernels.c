#include "kernels.h"

const char * get_program_file_name(kernel_type_t k) {
    static const char * file_names[] = {
        "src/native/opencl/add.cl",
        "src/native/opencl/add_scalar.cl",
        "src/native/opencl/add_bang.cl",
        "src/native/opencl/add_scalar_bang.cl"
    };

    return file_names[k];
}

const char * get_cl_function_name(kernel_type_t k) {
    static const char * module_names[] = {
        "add",
        "add_scalar",
        "add_bang",
        "add_scalar_bang"
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
    kernel = clCreateKernel(program, cl_function_name, &status);

    free((void*) file_contents);
    clReleaseProgram(program);
    return kernel;
}
