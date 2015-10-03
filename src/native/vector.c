#include "vector.h"

void gpu_matrix_release(vector * x) {
    // TODO!
}

size_t gpu_matrix_vector_datasize(vector * x) {
    return x->length * sizeof(double);
}

vector * gpu_matrix_vector_axpy(vector * v_x, double alpha, vector * v_y) {
    size_t global_work_size[1] = { v_x->length };
    size_t local_work_size[1] = { 100 };
    const size_t local_cache_size = sizeof(double) * (*local_work_size);
    vector * output;
    size_t datasize = gpu_matrix_vector_datasize(v_x);
    cl_int status;
    cl_command_queue cmd_queue = clCreateCommandQueue(
        context_get(),
        device_get(),
        0,
        &status
    );

    cl_mem buffer_x, buffer_y, buffer_output, buffer_local_cache;

    // Allocate memory for output vector object

    output = malloc(sizeof(vector));
    output->data = malloc(sizeof(double) * v_x->length);
    output->length = v_x->length;
    output->stride = 1;

    cl_kernel kernel = kernels_get(
        context_get(),
        device_get(),
        KERNEL_VECTOR_AXPY
    );
    buffer_x = buffers_create(
        CL_MEM_READ_ONLY,
        datasize,
        NULL,
        &status
    );
    buffer_y = buffers_create(
        CL_MEM_READ_ONLY,
        datasize,
        NULL,
        &status
    );
    buffer_output = buffers_create(
        CL_MEM_WRITE_ONLY,
        datasize,
        NULL,
        &status
    ); 
    status = clEnqueueWriteBuffer(
        cmd_queue, buffer_x,
        CL_FALSE, 0, datasize,
        v_x->data, 0, NULL, NULL
    );
    status = clEnqueueWriteBuffer(
        cmd_queue, buffer_y,
        CL_FALSE, 0, datasize,
        v_y->data, 0, NULL, NULL
    );

    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_x);
    status |= clSetKernelArg(kernel, 1, sizeof(index_t), &v_x->length);
    status |= clSetKernelArg(kernel, 2, sizeof(index_t), &v_x->stride);
    status |= clSetKernelArg(kernel, 3, sizeof(double), &alpha);
    status |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &buffer_y);
    status |= clSetKernelArg(kernel, 5, sizeof(index_t), &v_y->length);
    status |= clSetKernelArg(kernel, 6, sizeof(index_t), &v_y->stride);
    status |= clSetKernelArg(kernel, 7, local_cache_size, NULL);
    status |= clSetKernelArg(kernel, 8, local_cache_size, NULL);
    status |= clSetKernelArg(kernel, 9, sizeof(cl_mem), &buffer_output);
    status = clEnqueueNDRangeKernel(
        cmd_queue,
        kernel,
        1,
        NULL,
        global_work_size,
        local_work_size,
        0,
        NULL,
        NULL
    );
    clEnqueueReadBuffer(
        cmd_queue,
        buffer_output,
        CL_TRUE,
        0,
        datasize,
        output->data,
        0,
        NULL,
        NULL
    );

    return output;
}


