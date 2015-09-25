// Current model assumes that there is only 1 GPU
// So there is only 1 Global context throughout the program

#include "cl_helper.h"

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

    TRY_AND_CATCH_ERROR(
        status = clGetPlatformIDs(0, NULL, &num_platforms);,
        status
    )
    platforms = (cl_platform_id*) malloc(num_platforms * sizeof(cl_platform_id));
    status = clGetPlatformIDs(num_platforms, platforms, NULL);
    PLATFORM = platforms[0];

    // 2. Discover and initialize devices
    cl_uint num_devices = 0;
    cl_device_id * devices = NULL;
    TRY_AND_CATCH_ERROR(
        status = clGetDeviceIDs(
            platforms[0],
            CL_DEVICE_TYPE_ALL,
            0,
            NULL,
            &num_devices
        );,
        status
    );
    devices = (cl_device_id*) malloc(num_platforms * sizeof(cl_device_id));
    TRY_AND_CATCH_ERROR(
        status = clGetDeviceIDs(
            platforms[0],
            CL_DEVICE_TYPE_ALL,
            num_devices,
            devices,
            NULL
        );,
        status
    );
    DEVICE = devices[0];

    // 3. Create a context
    TRY_AND_CATCH_ERROR(
        CONTEXT = clCreateContext(
            NULL,
            num_devices,
            devices,
            NULL,
            NULL,
            &status
        );,
        status
    );
    
    is_init = 1;

    free(platforms);
    free(devices);
}

