#include "vector.h"

#define ENABLE_PROFILING 0

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

void describe_event(const char * description, cl_ulong time) {
    printf("%s: %.3f milliseconds\n", description, ((double) time * 0.000001));
}

double gpu_matrix_vector_asum(vector * v_x) {
    size_t datasize = gpu_matrix_vector_datasize(v_x);
    cl_int status;
    double output;
    cl_mem buffer_data;
    cl_command_queue cmd_queue;
    cl_kernel kernel;
    index_t remaining_length = v_x->length;
    cl_event write_buffer_event, enqueue_event, read_buffer_event;
#ifdef ENABLE_PROFILING
    cl_ulong write_buffer_start, write_buffer_end,
             enqueue_start, enqueue_end,
             read_buffer_start, read_buffer_end, total_time;
    clock_t wall_clock_time = 0;
    total_time = 0;
    wall_clock_time = clock();

#endif
    
    cmd_queue = clCreateCommandQueue(
        context_get(),
        device_get(),
        CL_QUEUE_PROFILING_ENABLE,
        &status
    );
    buffer_data = buffers_create(
        CL_MEM_READ_WRITE,
        datasize,
        NULL,
        &status
    );
    status = clEnqueueWriteBuffer(
        cmd_queue, buffer_data,
        CL_TRUE, 0, datasize,
        v_x->data, 0, NULL, &write_buffer_event 
    );

    // Print write buffer profilling information
#ifdef ENABLE_PROFILING
    clGetEventProfilingInfo(write_buffer_event,
        CL_PROFILING_COMMAND_START,
        sizeof(cl_ulong),
        &write_buffer_start,
        NULL
    );
    clGetEventProfilingInfo(
        write_buffer_event,
        CL_PROFILING_COMMAND_END,
        sizeof(cl_ulong),
        &write_buffer_end,
        NULL
    );
    describe_event("clEnqueueWriteBuffer : ", write_buffer_end - write_buffer_start);
    total_time += write_buffer_end - write_buffer_start;
#endif

    // Then, compute the sum 
    kernel = kernels_get(
        context_get(),
        device_get(),
        KERNEL_VECTOR_ASUM
    );

    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_data);
    status |= clSetKernelArg(kernel, 2, sizeof(index_t), &v_x->stride);

    while(remaining_length > 1) {
        size_t global_work_size[1] = { remaining_length / 2 };
        status = clSetKernelArg(kernel, 1, sizeof(index_t), &remaining_length);
        status = clEnqueueNDRangeKernel(
            cmd_queue,
            kernel,
            1,
            NULL,
            global_work_size,
            NULL,
            0,
            NULL,
            &enqueue_event 
        );
        remaining_length = (remaining_length+1) >> 1;
        clWaitForEvents(1, &enqueue_event);

#ifdef ENABLE_PROFILING
        clGetEventProfilingInfo(
            enqueue_event,
            CL_PROFILING_COMMAND_START,
            sizeof(cl_ulong),
            &enqueue_start,
            NULL
        );
        clGetEventProfilingInfo(
            enqueue_event,
            CL_PROFILING_COMMAND_END,
            sizeof(cl_ulong),
            &enqueue_end,
            NULL
        );
        describe_event("clEnqueueNDRangeKernel ", enqueue_end - enqueue_start);
        total_time += enqueue_end - enqueue_start;
#endif
        clReleaseEvent(enqueue_event);
    }

    clEnqueueReadBuffer(
        cmd_queue,
        buffer_data,
        CL_TRUE,
        0,
        sizeof(double),
        &output,
        0,
        NULL,
        &read_buffer_event 
    );

#ifdef ENABLE_PROFILING
    clGetEventProfilingInfo(
        read_buffer_event,
        CL_PROFILING_COMMAND_START,
        sizeof(cl_ulong),
        &read_buffer_start,
        NULL
    );
    clGetEventProfilingInfo(
        read_buffer_event,
        CL_PROFILING_COMMAND_END,
        sizeof(cl_ulong),
        &read_buffer_end,
        NULL
    );
    describe_event("clEnqueueReadBuffer : ", read_buffer_end - read_buffer_start);
    total_time += read_buffer_end - read_buffer_start;

    printf("Total HW Time : %.2f milliseconds\n", ((double) total_time) * 0.000001);
#endif

    clReleaseMemObject(buffer_data);
    clReleaseEvent(read_buffer_event);
    clReleaseEvent(write_buffer_event);

#ifdef ENABLE_PROFILING
    printf("Wall clock time: %f milliseconds\n",
        ((float) clock() - wall_clock_time )/CLOCKS_PER_SEC*1000 );

#endif

    return output;
}
