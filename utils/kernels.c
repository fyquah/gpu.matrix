// Current model assumes that there is only 1 GPU
// So there is only 1 Global context throughout the program

#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include "files.h"
#include "CL/cl.h"
#include "kernels.h"

bool is_init = false;
cl_platform_id PLATFORM;
cl_device_id DEVICE;
cl_context CONTEXT;

const char * get_program_file_name(const char * module_name) {
    init();
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

cl_command_queue command_queue_create(cl_command_queue_properties properties, cl_int * errcode_res) {
    return clCreateCommandQueue(CONTEXT, DEVICE, properties, errcode_res);
}

cl_mem buffers_create(cl_mem_flags flags, size_t size, void * host_ptr, cl_int * errcode_res) {
    return clCreateBuffer(CONTEXT, flags, size, host_ptr, errcode_res);
}

cl_kernel kernels_get(const char * module_name) {
    init();
    cl_int status;
    cl_kernel kernel;
    cl_program program;
    const char * filename = get_program_file_name(module_name);
    const char * file_contents = slurp(filename);

    program = clCreateProgramWithSource(
        CONTEXT, 1, (const char **) &file_contents,
        NULL, &status
    );
    status = clBuildProgram(
        program, 1, &DEVICE, NULL, NULL, NULL
    );

    kernel = clCreateKernel(program, module_name, &status);

    free((void*) file_contents);
    clReleaseProgram(program);
    return kernel;
}

// Frees all the objects' memory
void destroy() {
    
}

void init() {
    if (is_init) { 
        return;
    }
    cl_int status;
    // 1. Discover, and initialize platforms
    cl_uint num_platforms = 0;
    cl_platform_id * platforms = NULL;

    status = clGetPlatformIDs(0, NULL, &num_platforms);
    platforms = (cl_platform_id*) malloc(num_platforms * sizeof(cl_platform_id));
    status = clGetPlatformIDs(num_platforms, platforms, NULL);
    PLATFORM = platforms[0];

    // 2. Discover and initialize devices
    cl_uint num_devices = 0;
    cl_device_id * devices = NULL;
    status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
    devices = (cl_device_id*) malloc(num_platforms * sizeof(cl_device_id));
    status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
    DEVICE = devices[0];

    // 3. Create a context
    CONTEXT = clCreateContext(
        NULL,
        num_devices,
        devices,
        NULL,
        NULL,
        &status
    );
    
    is_init = 1;
}
