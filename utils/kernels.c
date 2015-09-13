#include "kernels.h"

const char * get_program_file_name(const char * module_name) {
    static const char * file_names[] = {
        "opencl/add.cl", "opencl/add_scalar.cl" };
    int index = -1;

    if(strcmp(module_name, "add") == 0) {
        index = 0;
    } else if (strcmp(module_name, "add_scalar") == 0) {
        index = 1;
    }

    return file_names[index];
}

cl_kernel kernels_get(cl_context context, cl_device_id device, const char * module_name) {
    cl_int status;
    cl_kernel kernel;
    cl_program program;
    const char * filename = get_program_file_name(module_name);
    const char * file_contents = slurp(filename);

    program = clCreateProgramWithSource(
        context, 1, (const char **) &file_contents,
        NULL, &status
    );
    status = clBuildProgram(
        program, 1, &device, NULL, NULL, NULL
    );

    kernel = clCreateKernel(program, module_name, &status);

    free((void*) file_contents);
    clReleaseProgram(program);
    return kernel;
}
