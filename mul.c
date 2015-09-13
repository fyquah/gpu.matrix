#include "mul.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <CL/cl.h>
// m_x is matrix x
// dims_x is dimension of matrix x
// m_y is matrix y
// dims_y is dimension of matrix y
// we want to find the product of m_x * m_y

void mul() {
    int *A, *B, *C, *C_cpu;
    const int A_c = 1000;
    const int A_r = 10000;
    const int B_c = 1000;
    const int B_r = A_c;
    const int C_c = B_c;
    const int C_r = A_r;
    const int elem_size = sizeof(int);
    const unsigned A_size = A_r * A_c * elem_size;
    const unsigned B_size = B_r * B_c * elem_size;
    const unsigned C_size = C_r * C_c * elem_size;
    const char * filename = "opencl/mul.cl";
    size_t global_work_size[2];
    size_t local_work_size[2];
    cl_int status;
    cl_context context = NULL;
    cl_command_queue cmd_queue = NULL;
    cl_kernel kernel = NULL;
    cl_mem buffer_a, buffer_b, buffer_c;

    // 0. initialize data
    A = (int*) malloc(A_r * A_c * elem_size);
    B = (int*) malloc(B_r * B_c * elem_size);
    C = (int*) malloc(C_r * C_c * elem_size);
    C_cpu = (int*) malloc(C_size);

    for (int i = 0 ; i < A_r ; i++) {
        for (int j = 0 ; j < A_c ; j++) {
            A[i*A_c+j] = rand() % 2000;
        }
    }

    for (int i = 0 ; i < B_r ; i++) {
        for (int j = 0 ; j < B_c ; j++) {
            B[i*B_c+j] = rand() % 2000;
        }
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
            A_size, NULL, &status);
    buffer_b = clCreateBuffer(context, CL_MEM_READ_ONLY, 
            B_size, NULL, &status);
    buffer_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
            C_size, NULL, &status);
    
    // 6. Write host data to device buffers
    status = clEnqueueWriteBuffer(
        cmd_queue, 
        buffer_a,
        CL_FALSE,
        0,
        A_size,
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
        B_size,
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
    } else {
        puts("Program compilation successful!");
    }

    // 8. Create the kernel

    kernel = clCreateKernel(program, "mul", &status);

    // 9. Set the kernel arguments
    status =  clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_a);
    status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer_b);
    status |= clSetKernelArg(kernel, 2, sizeof(cl_int), (void*) &A_c);
    status |= clSetKernelArg(kernel, 3, sizeof(cl_int), (void*) &B_c);
    status |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &buffer_c);

    // 10. COnfigure the work item structure
    global_work_size[0] = C_r;
    global_work_size[1] = C_c;

    local_work_size[0] = 10;
    local_work_size[1] = 10;
    
    // 11. Enqueue the kernel for execution
    
    status = clEnqueueNDRangeKernel(cmd_queue, kernel, 2, NULL,
           global_work_size, local_work_size, 0, NULL, NULL);
    
    // 12. Read the output buffer back to the host
    clock_t d = clock(); 
    clEnqueueReadBuffer(cmd_queue,buffer_c, CL_TRUE, 0, C_size, C ,0, NULL, NULL);
    d = clock() - d;
    printf("Ran in %f seconds\n", ((float) d)/CLOCKS_PER_SEC);

    // 13. Compare the results for accuracy
    d = clock(); 
    for (int i = 0 ; i < C_r ; i++) {
        for (int j = 0 ; j < C_c ; j++) {
            int sum = 0;

            for (int k = 0 ; k < A_c ; k++) {
                sum += A[i*A_c+k] * B[k*B_c+j];
            }

            C_cpu[i*C_c+j] = sum;
        }
    }
    d = clock() - d;
    printf("Ran in %f seconds\n", ((float) d)/CLOCKS_PER_SEC);
   

    // 14. Free memory objects
    for (int i = 0 ; i < C_r ; i++) {
        for (int j = 0 ; j < C_c ; j++) {
            if (C_cpu[i*C_c+j] != C[i*C_c+j]) {
                printf("WRONG!\n");
            }
        }
    }
}
