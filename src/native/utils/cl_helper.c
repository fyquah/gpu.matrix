// Current model assumes that there is only 1 GPU
// So there is only 1 Global context throughout the program

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
void gpu_matrix_destroy() {
    clReleaseContext(CONTEXT);    
}

void gpu_matrix_init() {
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

    free(platforms);
    free(devices);
}

