// Current model assumes that there is only 1 GPU
// So there is only 1 Global context throughout the program

#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include "files.h"
#include "CL/cl.h"
#include "cl_helper.h"

bool is_init = false;
cl_platform_id PLATFORM;
cl_device_id DEVICE;
cl_context CONTEXT;

cl_command_queue command_queue_create(cl_command_queue_properties properties, cl_int * errcode_res) {
    return clCreateCommandQueue(CONTEXT, DEVICE, properties, errcode_res);
}

cl_mem buffers_create(cl_mem_flags flags, size_t size, void * host_ptr, cl_int * errcode_res) {
    return clCreateBuffer(CONTEXT, flags, size, host_ptr, errcode_res);
}

cl_context context_get() {
    return CONTEXT;
}

cl_device_id device_get() {
    return DEVICE;
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

opencl_config_t get_common_config () {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue cmd_queue;
    cl_int status;
    
    // 1. Discover, and initialize platforms
    cl_uint num_platforms = 0;
    cl_platform_id * platforms = NULL;

    status = clGetPlatformIDs(0, NULL, &num_platforms);
    platforms = (cl_platform_id*) malloc(num_platforms * sizeof(cl_platform_id));
    status = clGetPlatformIDs(num_platforms, platforms, NULL);
    platform = platforms[0];

    // 2. Discover and initialize devices
    cl_uint num_devices = 0;
    cl_device_id * devices = NULL;
    status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
    devices = (cl_device_id*) malloc(num_platforms * sizeof(cl_device_id));
    status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
    device = devices[0];

    // 3. Create a context
    context = clCreateContext(
        NULL,
        num_devices,
        devices,
        NULL,
        NULL,
        &status
    );
    
    // 4. Create a command queue
    cmd_queue = clCreateCommandQueue(context, devices[0], 0, &status);

    // TODO: Free object memory
    // TODO: Handle object construction errors
    
    opencl_config_t config = { context, platform, device, cmd_queue };
    return config;
}
