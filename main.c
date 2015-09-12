// standard libraries
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// OpenCL
#include <CL/cl.h>

char * slurp(const char * filename) {
    FILE * program_handle;
    unsigned long program_size;
    char * program_buffer;

    program_handle = fopen(filename, "r");
    fseek(program_handle, 0, SEEK_END);
    program_size = ftell(program_handle);
    rewind(program_handle);

    program_buffer = (char*) malloc(program_size + 1);
    program_buffer[program_size] = '\0';
    fread(program_buffer, sizeof(char), program_size, program_handle);
    fclose(program_handle);

    return program_buffer;
}


int main(int args, char ** argv) {
    assert(args == 2);
    int *A , *B, *C;
    const int n_elements = 10000;
    size_t datasize = sizeof(int) * n_elements;
    size_t global_work_size[1];
    const char * filename = argv[1];
    cl_int status;
    cl_context context = NULL;
    cl_command_queue cmd_queue = NULL;
    cl_kernel kernel = NULL;
    cl_mem buffer_a, buffer_b, buffer_c;
    
    A = (int*) malloc(datasize);
    B = (int*) malloc(datasize);
    C = (int*) malloc(datasize);
    
    for (int i = 0 ; i < n_elements ; i++) {
        A[i] = i;
        B[i] = i * 2;
    }

    // 1. Discover, and initialize platforms
    cl_uint num_platforms = 0;
    cl_platform_id * platforms = NULL;

    status = clGetPlatformIDs(0, NULL, &num_platforms);
    platforms = (cl_platform_id*) malloc(num_platforms * sizeof(cl_platform_id));
    status = clGetPlatformIDs(num_platforms, platforms, NULL);

    // 2. Discover and initialize devices
    cl_uint num_devices = 0;
    cl_device_id * devices = NULL;
    status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
    devices = (cl_device_id*) malloc(num_platforms * sizeof(cl_device_id));
    status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);

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
    
    // 5. Create device buffers
    buffer_a = clCreateBuffer(context, CL_MEM_READ_ONLY,
            datasize, NULL, &status);
    buffer_b = clCreateBuffer(context, CL_MEM_READ_ONLY, 
            datasize, NULL, &status);
    buffer_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
            datasize, NULL, &status);
    
    // 6. Write host data to device buffers
    status = clEnqueueWriteBuffer(
        cmd_queue, 
        buffer_a,
        CL_FALSE,
        0,
        datasize,
        A,
        0,
        NULL,
        NULL
    );
    status = clEnqueueWriteBuffer(
        cmd_queue, 
        buffer_b,
        CL_FALSE,
        0,
        datasize,
        B,
        0,
        NULL,
        NULL
    );


    // 7. Load and compile program
    const char * contents = slurp(filename);
    cl_program program = clCreateProgramWithSource(
        context,
        1,
        (const char **) &contents,
        NULL,
        &status
    );
    status = clBuildProgram(
        program,
        num_devices,
        devices,
        NULL,
        NULL,
        NULL
    );

    if (status == CL_BUILD_PROGRAM_FAILURE) {
        size_t log_size;
        clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

        char *log = (char *) malloc(log_size);

        clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

        printf("%s\n", log);
    }
    // 8. Create the kernel

    kernel = clCreateKernel(program, "vecadd", &status);

    // 9. Set the kernel arguments
    
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_a);
    status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer_b);
    status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &buffer_c);

    // 10. COnfigure the work item structure
    
    global_work_size[0] = n_elements;

    // 11. Enqueue the kernel for execution
    
    status = clEnqueueNDRangeKernel(cmd_queue, kernel, 1, NULL,
           global_work_size, NULL, 0, NULL, NULL);

    // 12. Read the output buffer back to the host
   
    clEnqueueReadBuffer(cmd_queue, buffer_c, CL_TRUE, 0, datasize, C ,0, NULL, NULL);

}
