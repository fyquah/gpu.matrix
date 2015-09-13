#include <CL/cl.h>
#include <stdbool.h>
#include <stdio.h>

#include "ndarray.h"
#include "utils.h"

void ndarray_release(ndarray * arr) {
    free(arr->data);
    free(arr->shape);
    free(arr->strides);
    free(arr);
}

unsigned ndarray_elements_count(const ndarray * arr) {
    int n_elements = 1;
    for (int i = 0 ; i < arr->ndims ; i++) {
        n_elements *= arr->shape[i];
    }

    return n_elements;
}

unsigned ndarray_datasize(const ndarray * arr) {
    return ndarray_elements_count(arr) * sizeof(double);
}

ndarray * ndarray_clone_structure(ndarray * arr_x) {
    const unsigned datasize = ndarray_datasize(arr_x);
    output * ndarray = malloc(sizeof(ndarray));
    
    output->data     = (double*) malloc(datasize);
    output->strides  = array_size_t_copy(arr_x->strides, (unsigned long long) arr_x->ndims);
    output->shape    = array_size_t_copy(arr_x->shape, arr_x->ndims);
    output->ndims    = arr_x->ndims;

    return output;
}

// basic ops
ndarray * ndarray_add_scalar(ndarray * arr_x, double y) {
    const size_t * shape = arr_x->shape;
    const unsigned datasize = ndarray_datasize(arr_x);
    cl_kernel kernel;
    cl_int status;
    cl_mem buffer_x, buffer_output;
    cl_command_queue cmd_queue;
    ndarray * output = ndarray_clone_structure(arr_x);
    size_t global_work_size[1];
    global_work_size[0] = ndarray_elements_count(arr_x);

    cmd_queue = command_queue_create(0, &status);
    output = (ndarray*) malloc(sizeof(ndarray));
    kernel = kernels_get(context_get(), device_get(), "add_scalar");
    buffer_x = buffers_create(CL_MEM_READ_ONLY, datasize, NULL, &status);
    buffer_output = buffers_create(CL_MEM_WRITE_ONLY, datasize, NULL, &status);
    status = clEnqueueWriteBuffer(cmd_queue, buffer_x, CL_FALSE, 0,
            ndarray_datasize(arr_x), arr_x->data, 0, NULL, NULL);
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*) &buffer_x);
    status |= clSetKernelArg(kernel, 1, sizeof(double), (void*) &y);
    status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*) &buffer_output);
    status = clEnqueueNDRangeKernel(cmd_queue, kernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL);
    
    clEnqueueReadBuffer(cmd_queue, buffer_output, CL_TRUE, 0, ndarray_datasize(output), output->data, 0, NULL, NULL);

    return output;
}

ndarray * ndarray_add(const ndarray * arr_x, ndarray * arr_y){
    const size_t * shape = arr_x->shape;
    const size_t * global_work_size = arr_x->shape; 
    char * program_file_contents;
    cl_mem buffer_x, buffer_y, buffer_output;
    cl_int status;
    cl_program program;
    cl_kernel kernel;
    unsigned * strides;
    ndarray * output = (ndarray*) malloc(sizeof(ndarray));

    // 1. Get common openCL configuration
    opencl_config_t common = get_common_config();
    cl_context context = common.context;
    cl_platform_id platform = common.platform;
    cl_device_id device = common.device;
    cl_command_queue cmd_queue = common.cmd_queue;

    // 2. resize strides if needed 
    // TODO: Reshape stride

    // 3. Create device buffers
    buffer_x = clCreateBuffer(context, CL_MEM_READ_ONLY,
            ndarray_datasize(arr_x), NULL, &status);
    buffer_y = clCreateBuffer(context, CL_MEM_READ_ONLY,
            ndarray_datasize(arr_y), NULL, &status);
    buffer_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
            ndarray_datasize(arr_x), NULL, &status);

    // 4. Write host data to device buffers
    status = clEnqueueWriteBuffer(cmd_queue, buffer_x, CL_FALSE,
            0, ndarray_datasize(arr_x), arr_x->data, 0, NULL, NULL);
    status = clEnqueueWriteBuffer(cmd_queue, buffer_y, CL_FALSE,
            0, ndarray_datasize(arr_y), arr_y->data, 0, NULL, NULL);

    // 5. Load and compile program
    program_file_contents = slurp("opencl/add.cl");
    program = clCreateProgramWithSource(
        context, 1, (const char **) &program_file_contents, 
        NULL, &status
    );
    status = clBuildProgram(
        program, 1, &device, NULL, NULL, NULL
    );

    // 6. Create the kernel
    kernel = clCreateKernel(program, "add", &status);

    // 7. Set the kernel arguments
    status  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_x);
    status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer_y);
    status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &buffer_output);

    // 8. Enqueue the kernel for execution
    status = clEnqueueNDRangeKernel(cmd_queue, kernel, 1, NULL,
            global_work_size, NULL, 0, NULL, NULL);

    // 9. Read the otuput back to the host
    output->data     = (double*) malloc(ndarray_datasize(arr_x));
    output->strides  = array_size_t_copy(arr_x->strides, (unsigned long long) arr_x->ndims);
    output->shape    = array_size_t_copy(arr_x->shape, arr_x->ndims);
    output->ndims    = arr_x->ndims;

    // 10. Write the output back
    clEnqueueReadBuffer(cmd_queue, buffer_output, CL_TRUE, 0, ndarray_datasize(output), 
            output->data, 0, NULL, NULL);

    // 11. free object memory
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(cmd_queue);
    clReleaseMemObject(buffer_x);
    clReleaseMemObject(buffer_y);
    clReleaseMemObject(buffer_output);
    clReleaseContext(context);
    free(program_file_contents);

    return output;
}

