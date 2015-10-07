#include "vector.h"

void gpu_matrix_vector_buffer_axpy_BANG(
    vector_buffer * v_x,
    double d,
    vector_buffer * v_y,
    cl_command_queue cmd_queue
) {
    
    size_t global_work_size[1] = { v_x->length };
    size_t datasize = v_x->datasize;
    cl_int status;
    cl_event read_buffer_events[1], enqueue_events[1];

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
    clReleaseEvent(enqueue_events[0]);
}

void gpu_matrix_vector_buffer_asum_BANG(
    vector_buffer * v_x,
    cl_command_queue cmd_queue
){
    cl_event enqueue_events[1];
    cl_kernel kernel;
    cl_int status;
    index_t remaining_length;

    // Then, compute the sum 
    kernel = kernels_get(
        context_get(),
        device_get(),
        KERNEL_VECTOR_ASUM
    );

    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &v_x->buffer);
    status |= clSetKernelArg(kernel, 2, sizeof(index_t), &v_x->stride);

    remaining_length = v_x->length;
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
        clReleaseEvent(enqueue_events[0]);
    }
}

void gpu_matrix_vector_buffer_mul_BANG(
    vector_buffer * v_x,
    vector_buffer * v_y,
    cl_command_queue cmd_queue
) {
    size_t global_work_size[1];
    cl_event enqueue_events[1];
    cl_kernel kernel;
    cl_int status;
    index_t remaining_length;

    // Then, compute the sum 
    kernel = kernels_get(
        context_get(),
        device_get(),
        KERNEL_VECTOR_MUL_BANG 
    );
    global_work_size[0] = v_x->length;

    status  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &v_x->buffer);
    status |= clSetKernelArg(kernel, 1, sizeof(index_t), &v_x->length);
    status |= clSetKernelArg(kernel, 2, sizeof(index_t), &v_x->stride);
    status |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &v_y->buffer);
    status |= clSetKernelArg(kernel, 4, sizeof(index_t), &v_y->length);
    status |= clSetKernelArg(kernel, 5, sizeof(index_t), &v_y->stride);

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

    clWaitForEvents(1, enqueue_events);
    clReleaseEvent(enqueue_events[0]);
}

void gpu_matrix_vector_buffer_square_BANG(
    vector_buffer * v_x,
    cl_command_queue cmd_queue
) {
    size_t global_work_size[1];
    cl_event enqueue_events[1];
    cl_kernel kernel;
    cl_int status;

    global_work_size[0] = v_x->length;
    kernel = kernels_get(
        context_get(),
        device_get(),
        KERNEL_VECTOR_SQUARE_BANG
    );
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &v_x->buffer);
    status = clSetKernelArg(kernel, 1, sizeof(index_t), &v_x->length);
    status |= clSetKernelArg(kernel, 2, sizeof(index_t), &v_x->stride);

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
    clWaitForEvents(1, enqueue_events);
    clReleaseEvent(enqueue_events[0]);
}

void gpu_matrix_vector_buffer_rot_BANG(
    vector_buffer* v_x,
    vector_buffer* v_y,
    double c,
    double s,
    cl_command_queue cmd_queue
){
    size_t global_work_size[1];
    cl_event enqueue_events[1];
    cl_kernel kernel;
    cl_int status;

    global_work_size[0] = v_x->length;
    kernel = kernels_get(
        context_get(),
        device_get(),
        KERNEL_VECTOR_ROT_BANG
    );
    status  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &v_x->buffer);
    status |= clSetKernelArg(kernel, 1, sizeof(index_t), &v_x->length);
    status |= clSetKernelArg(kernel, 2, sizeof(index_t), &v_x->stride);
    status |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &v_y->buffer);
    status |= clSetKernelArg(kernel, 4, sizeof(index_t), &v_y->length);
    status |= clSetKernelArg(kernel, 5, sizeof(index_t), &v_y->stride);
    status |= clSetKernelArg(kernel, 6, sizeof(double), &c);
    status |= clSetKernelArg(kernel, 7, sizeof(double), &s);
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

    clWaitForEvents(1, enqueue_events);
    clReleaseEvent(enqueue_events[0]);
}

void gpu_matrix_vector_buffer_abs_BANG(
    vector_buffer * v_x,
    cl_command_queue cmd_queue
) {
    size_t global_work_size[1];
    cl_event enqueue_events[1];
    cl_kernel kernel;
    cl_int status;

    global_work_size[0] = v_x->length;
    kernel = kernels_get(
        context_get(),
        device_get(),
        KERNEL_VECTOR_ABS_BANG
    );
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &v_x->buffer);
    status = clSetKernelArg(kernel, 1, sizeof(index_t), &v_x->length);
    status |= clSetKernelArg(kernel, 2, sizeof(index_t), &v_x->stride);

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
    clWaitForEvents(1, enqueue_events);
    clReleaseEvent(enqueue_events[0]);
}

void gpu_matrix_vector_buffer_max_BANG(
    vector_buffer* v_x,
    cl_command_queue cmd_queue
){
    size_t global_work_size[1];
    cl_event enqueue_events[1];
    cl_kernel kernel;
    cl_int status;
    index_t remaining_length;

    global_work_size[0] = v_x->length;
    kernel = kernels_get(
        context_get(),
        device_get(),
        KERNEL_VECTOR_MAX_BANG
    );
    status  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &v_x->buffer);
    status |= clSetKernelArg(kernel, 2, sizeof(index_t), &v_x->stride);

    remaining_length = v_x->length;
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
        clReleaseEvent(enqueue_events[0]);
    }
}

void gpu_matrix_vector_buffer_min_BANG(
    vector_buffer * v_x,
    cl_command_queue cmd_queue
){
    cl_event enqueue_events[1];
    cl_kernel kernel;
    cl_int status;
    index_t remaining_length;

    // Then, compute the sum 
    kernel = kernels_get(
        context_get(),
        device_get(),
        KERNEL_VECTOR_MIN_BANG
    );

    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &v_x->buffer);
    status |= clSetKernelArg(kernel, 2, sizeof(index_t), &v_x->stride);

    remaining_length = v_x->length;
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
        clReleaseEvent(enqueue_events[0]);
    }
}

cl_mem gpu_matrix_vector_buffer_imin(
    vector_buffer * v_x,
    cl_command_queue cmd_queue
){
    cl_event enqueue_events[1];
    cl_kernel kernel;
    cl_int status;
    index_t remaining_length;
    cl_mem buffer_indices;
    size_t range_global_work_size[1];

    range_global_work_size[0] = v_x->length;
    buffer_indices = buffers_create(
        CL_MEM_READ_WRITE,
        sizeof(index_t) * v_x->length,
        NULL,
        &status
    );

    kernel = kernels_get(
        context_get(),
        device_get(),
        KERNEL_VECTOR_RANGE
    );
    status  = clSetKernelArg(kernel, 0, sizeof(index_t), &v_x->length);
    status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer_indices);
    status = clEnqueueNDRangeKernel(
        cmd_queue,
        kernel,
        1,
        NULL,
        range_global_work_size,
        NULL,
        0,
        NULL,
        enqueue_events
    );

    clWaitForEvents(1, enqueue_events);

    // Create a range and index kernel firstkA
    clReleaseEvent(enqueue_events[0]);

    // Then, compute the sum 
    kernel = kernels_get(
        context_get(),
        device_get(),
        KERNEL_VECTOR_IMIN_BANG 
    );

    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &v_x->buffer);
    status |= clSetKernelArg(kernel, 2, sizeof(index_t), &v_x->stride);
    status |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &buffer_indices);

    remaining_length = v_x->length;
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
        clReleaseEvent(enqueue_events[0]);
    }

    return buffer_indices;
}

cl_mem gpu_matrix_vector_buffer_imax(
    vector_buffer * v_x,
    cl_command_queue cmd_queue
){
    cl_event enqueue_events[1];
    cl_kernel kernel;
    cl_int status;
    index_t remaining_length;
    cl_mem buffer_indices;
    size_t range_global_work_size[1];
   
    buffer_indices = buffers_create(
        CL_MEM_READ_WRITE,
        sizeof(index_t) * v_x->length,
        NULL,
        &status
    );
    kernel = kernels_get(
        context_get(),
        device_get(),
        KERNEL_VECTOR_RANGE
    );

    status = clSetKernelArg(kernel, 0, sizeof(index_t), &v_x->length);
    status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer_indices);

    status = clEnqueueNDRangeKernel(
        cmd_queue,
        kernel,
        1,
        NULL,
        range_global_work_size,
        NULL,
        0,
        NULL,
        enqueue_events
    );
    clWaitForEvents(1, enqueue_events);

    // Create a range and index kernel firstkA
    clReleaseEvent(enqueue_events[0]);

    // Then, compute the sum 
    kernel = kernels_get(
        context_get(),
        device_get(),
        KERNEL_VECTOR_IMAX_BANG 
    );

    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &v_x->buffer);
    status |= clSetKernelArg(kernel, 2, sizeof(index_t), &v_x->stride);
    status |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &buffer_indices);

    remaining_length = v_x->length;
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
        clReleaseEvent(enqueue_events[0]);
    }

    return buffer_indices;
}

