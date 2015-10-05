#include "vector.h"

const index_t ONE = 1;

void describe_event(const char * description, cl_ulong time) {
    printf("%s: %.3f milliseconds\n", description, ((double) time * 0.000001));
}

cl_ulong get_event_time(cl_event event, char * description) {
    cl_ulong end, start;

    clGetEventProfilingInfo(
        event,
        CL_PROFILING_COMMAND_START,
        sizeof(cl_ulong),
        &start,
        NULL
    );
    clGetEventProfilingInfo(
        event,
        CL_PROFILING_COMMAND_END,
        sizeof(cl_ulong),
        &end,
        NULL
    );

#ifdef ENABLE_PROFILING
    printf("%s: %.3f milliseconds\n", description, ((double) (end-start) * 0.000001));
#endif

    return end-start;
}

void gpu_matrix_release(vector * x) {
    // TODO, don't just free
}

size_t gpu_matrix_vector_datasize(vector * x) {
    return x->length * sizeof(double);
}

vector * gpu_matrix_vector_axpy(vector * v_x, double alpha, vector * v_y) {
    size_t global_work_size[1] = { v_x->length };
    vector * output;
    size_t datasize = gpu_matrix_vector_datasize(v_x);
    cl_int status;
    cl_command_queue cmd_queue = clCreateCommandQueue(
        context_get(),
        device_get(),
        CL_QUEUE_PROFILING_ENABLE,
        &status
    );
    cl_event write_buffer_events[2], read_buffer_events[1], enqueue_events[1];
    
#ifdef ENABLE_PROFILING
    cl_ulong write_buffer_x_start, write_buffer_x_end,
             write_buffer_y_start, write_buffer_y_end,
             enqueue_start, enqueue_end,
             read_buffer_start, read_buffer_end, total_time;
    clock_t wall_clock_time = 0;
    total_time = 0;
    wall_clock_time = clock();
#endif

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
        v_x->data, 0, NULL, write_buffer_events 
    );
    status = clEnqueueWriteBuffer(
        cmd_queue, buffer_y,
        CL_FALSE, 0, datasize,
        v_y->data, 0, NULL, write_buffer_events+1 
    );
    clWaitForEvents(2, write_buffer_events);

    status  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_x);
    status |= clSetKernelArg(kernel, 1, sizeof(index_t), &v_x->length);
    status |= clSetKernelArg(kernel, 2, sizeof(index_t), &v_x->stride);
    status |= clSetKernelArg(kernel, 3, sizeof(double), &alpha);
    status |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &buffer_y);
    status |= clSetKernelArg(kernel, 5, sizeof(index_t), &v_y->length);
    status |= clSetKernelArg(kernel, 6, sizeof(index_t), &v_y->stride);
    status |= clSetKernelArg(kernel, 7, sizeof(cl_mem), &buffer_output);
    status = clEnqueueNDRangeKernel(
        cmd_queue,
        kernel,
        1,
        NULL,
        global_work_size,
        NULL,
        0,
        NULL,
        enqueue_events
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
        read_buffer_events
    );
    clWaitForEvents(1, read_buffer_events);

#ifdef ENABLE_PROFILING
    cl_ulong start, end;
    clGetEventProfilingInfo(
        read_buffer_events[0],
        CL_PROFILING_COMMAND_START,
        sizeof(cl_ulong),
        &start,
        NULL
    );
    clGetEventProfilingInfo(
        read_buffer_events[0],
        CL_PROFILING_COMMAND_END,
        sizeof(cl_ulong),
        &end,
        NULL
    );

    total_time = 0.0;
    total_time += get_event_time(write_buffer_events[0], "clEnqueueWriteBuffer (buffer_x): ");
    total_time += get_event_time(write_buffer_events[1], "clEnqueueWriteBuffer (buffer_y): ");
    total_time += get_event_time(enqueue_events[0], "clEnqueueNDRangeKernel: ");
    total_time += get_event_time(read_buffer_events[0], "clEnqueueReadBuffer (buffer_output)");
    printf("HW Time milliseconds: %.3f\n", (total_time) * 0.000001);

#endif

    clReleaseEvent(write_buffer_events[0]);
    clReleaseEvent(write_buffer_events[1]);
    clReleaseEvent(enqueue_events[0]);
    clReleaseEvent(read_buffer_events[0]);

#ifdef ENABLE_PROFILING
    printf("Wall clock time: %f milliseconds\n",
        ((float) clock() - wall_clock_time )/CLOCKS_PER_SEC*1000 );
#endif

    return output;
}

void gpu_matrix_vector_swap(vector * v_x, vector * v_y) {
    vector tmp_x = *v_x;
    *v_x = *v_y;
    *v_y = tmp_x;
}

double gpu_matrix_vector_dot(vector * v_x, vector * v_y) {
    size_t datasize = gpu_matrix_vector_datasize(v_x);
    size_t global_work_size[1] = { v_x->length };
    cl_int status;
    double output;
    cl_mem buffer_data_output, buffer_data_x, buffer_data_y;
    cl_command_queue cmd_queue;
    cl_kernel kernel;
    index_t remaining_length = v_x->length;
    cl_event enqueue_events[1],
             write_buffer_events[2],
             read_buffer_events[1];
    
#ifdef ENABLE_PROFILING
    clock_t wall_clock_time = clock();
    cl_ulong total_time = 0;
#endif

    cmd_queue = clCreateCommandQueue(
        context_get(),
        device_get(),
        CL_QUEUE_PROFILING_ENABLE,
        &status
    );
    kernel = kernels_get(
        context_get(),
        device_get(),
        KERNEL_VECTOR_MUL
    );
    buffer_data_x = buffers_create(
        CL_MEM_READ_ONLY,
        datasize,
        NULL,
        &status
    );
    buffer_data_y = buffers_create(
        CL_MEM_READ_ONLY,
        datasize,
        NULL,
        &status
    );
    buffer_data_output = buffers_create(
        CL_MEM_READ_WRITE,
        datasize,
        NULL,
        &status
    );

    status = clEnqueueWriteBuffer(
        cmd_queue, buffer_data_x,
        CL_FALSE, 0, datasize,
        v_x->data, 0, NULL,
        write_buffer_events
    );
    status = clEnqueueWriteBuffer(
        cmd_queue, buffer_data_y,
        CL_FALSE, 0, datasize,
        v_y->data, 0, NULL,
        write_buffer_events+1
    );
    clWaitForEvents(2, write_buffer_events);

    // Compute the Interleave multiplication first
    status  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_data_x);
    status |= clSetKernelArg(kernel, 1, sizeof(index_t), &v_x->length);
    status |= clSetKernelArg(kernel, 2, sizeof(index_t), &v_x->stride);
    status |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &buffer_data_y);
    status |= clSetKernelArg(kernel, 4, sizeof(index_t), &v_y->length);
    status |= clSetKernelArg(kernel, 5, sizeof(index_t), &v_y->stride);
    status |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &buffer_data_output);
    status = clEnqueueNDRangeKernel(
        cmd_queue,
        kernel,
        1,
        NULL,
        global_work_size,
        NULL,
        0,
        NULL,
        enqueue_events
    );


#ifdef ENABLE_PROFILING

    total_time += get_event_time(write_buffer_events[0], "clEnqueueWriteBuffer (buffer_x)");
    total_time += get_event_time(write_buffer_events[1], "clEnqueueWriteBuffer (buffer_y)");
    total_time += get_event_time(enqueue_events[0], "clEnqueueNDRangeKernel (vector_mul)");

#endif

    // Then, compute the sum 
    kernel = kernels_get(
        context_get(),
        device_get(),
        KERNEL_VECTOR_ASUM
    );

    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_data_output);
    status |= clSetKernelArg(kernel, 2, sizeof(index_t), &ONE);

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
            enqueue_events
        );
        remaining_length = (remaining_length+1) >> 1;
        clWaitForEvents(1, enqueue_events);

#ifdef ENABLE_PROFILING
        total_time += get_event_time(enqueue_events[0], "clEnqueueNDRangeKernel (asum)");
#endif
        clReleaseEvent(enqueue_events[0]);
    }

    clEnqueueReadBuffer(
        cmd_queue,
        buffer_data_output,
        CL_TRUE,
        0,
        sizeof(double),
        &output,
        0,
        NULL,
        read_buffer_events
    );
    clWaitForEvents(1, read_buffer_events);

#ifdef ENABLE_PROFILING
    total_time += get_event_time(read_buffer_events[0], "clEnqueueReadBuffer");
#endif

    // Release buffer objects
    clReleaseMemObject(buffer_data_output);
    clReleaseEvent(read_buffer_events[0]);
    clReleaseEvent(write_buffer_events[0]);
    clReleaseEvent(write_buffer_events[1]);

#ifdef ENABLE_PROFILING
    printf("Total HW Time: %.3f milliseconds \n", total_time * 0.000001);
    printf("Wall clock time: %f milliseconds\n",
        ((float) clock() - wall_clock_time )/CLOCKS_PER_SEC*1000 );
#endif

    return output;
}

double gpu_matrix_vector_asum(vector * v_x) {
    size_t datasize = gpu_matrix_vector_datasize(v_x);
    cl_int status;
    double output;
    cl_mem buffer_data;
    cl_command_queue cmd_queue;
    cl_kernel kernel;
    index_t remaining_length = v_x->length;
    cl_event write_buffer_events[1],
             enqueue_events[1],
             read_buffer_events[1];

#ifdef ENABLE_PROFILING
    clock_t wall_clock_time = clock();
    cl_ulong total_time = 0;
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
        v_x->data, 0, NULL, write_buffer_events
    );

    // Print write buffer profilling information
#ifdef ENABLE_PROFILING
    total_time += get_event_time(write_buffer_events[0], "clEnqueueWriteBuffer:");
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
            enqueue_events 
        );
        remaining_length = (remaining_length+1) >> 1;
        clWaitForEvents(1, enqueue_events);

#ifdef ENABLE_PROFILING
        total_time += get_event_time(enqueue_events[0], "clEnqueueNDRangeKernel:");
#endif
        clReleaseEvent(enqueue_events[0]);
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
        read_buffer_events
    );

#ifdef ENABLE_PROFILING
    total_time += get_event_time(read_buffer_events[0], "clEnqueueReadBuffer:");
#endif

    clReleaseMemObject(buffer_data);
    clReleaseEvent(read_buffer_events[0]);
    clReleaseEvent(write_buffer_events[0]);

#ifdef ENABLE_PROFILING
    printf("Total HW Time : %.2f milliseconds\n", ((double) total_time) * 0.000001);
    printf("Wall clock time: %f milliseconds\n",
        ((float) clock() - wall_clock_time )/CLOCKS_PER_SEC*1000 );

#endif

    return output;
}
