#include "vector.h"

void gpu_matrix_vector_buffer_axpy_BANG(vector_buffer * v_x, double d, vector_buffer * v_y) {
    
    size_t global_work_size[1] = { v_x->length };
    size_t datasize = v_x->datasize;
    cl_int status;
    cl_event read_buffer_events[1], enqueue_events[1];
    cl_command_queue cmd_queue = clCreateCommandQueue(
        context_get(),
        device_get(),
        CL_QUEUE_PROFILING_ENABLE,
        &status
    );

#ifdef ENABLE_PROFILING
    cl_ulong write_buffer_x_start, write_buffer_x_end,
             write_buffer_y_start, write_buffer_y_end,
             enqueue_start, enqueue_end,
             read_buffer_start, read_buffer_end, total_time;
    clock_t wall_clock_time = 0;
    total_time = 0;
    wall_clock_time = clock();
#endif

    // Allocate memory for output vector object
    cl_kernel kernel = kernels_get(
        context_get(),
        device_get(),
        KERNEL_VECTOR_AXPY_BANG
    );

    status  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &v_x->buffer);
    status |= clSetKernelArg(kernel, 1, sizeof(index_t), &v_x->length);
    status |= clSetKernelArg(kernel, 2, sizeof(index_t), &v_x->stride);
    status |= clSetKernelArg(kernel, 3, sizeof(double), &d);
    status |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &v_y->buffer);
    status |= clSetKernelArg(kernel, 5, sizeof(index_t), &v_y->length);
    status |= clSetKernelArg(kernel, 6, sizeof(index_t), &v_y->stride);
    status |= clEnqueueNDRangeKernel(
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

    clWaitForEvents(1, enqueue_events);

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
    total_time += get_event_time(enqueue_events[0], "clEnqueueNDRangeKernel: ");
    total_time += get_event_time(read_buffer_events[0], "clEnqueueReadBuffer (buffer_output)");
    printf("HW Time milliseconds: %.3f\n", (total_time) * 0.000001);

#endif

    clReleaseEvent(enqueue_events[0]);

#ifdef ENABLE_PROFILING
    printf("Wall clock time: %f milliseconds\n",
        ((float) clock() - wall_clock_time )/CLOCKS_PER_SEC*1000 );
#endif

    
}
