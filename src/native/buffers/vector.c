#include "vector.h"

static inline void gpu_matrix_vector_buffer_map_BANG_helper(
    unsigned vector_arg_count,
    vector_buffer * arr_vector_buffer[],
    unsigned scalar_arg_count,
    double arr_scalar[], 
    cl_command_queue cmd_queue,
    kernel_type_t kernel_id
){

    size_t global_work_size[1];
    size_t datasize = arr_vector_buffer[0]->datasize;
    cl_int status;
    cl_event read_buffer_events[1], enqueue_events[1];

    // Allocate memory for output vector object
    // assume there is at least one vector!
    global_work_size[0] = arr_vector_buffer[0]->length;
    cl_kernel kernel = kernels_get(
        context_get(),
        device_get(),
        kernel_id 
    );

    for (unsigned i = 0 ; i < vector_arg_count ; i++) {
        vector_buffer * v_x = arr_vector_buffer[i];
        status  = clSetKernelArg(kernel, i * 3, sizeof(cl_mem), &v_x->buffer);
        status |= clSetKernelArg(kernel, i * 3 + 1, sizeof(index_t), &v_x->length);
        status |= clSetKernelArg(kernel, i * 3 + 2, sizeof(index_t), &v_x->stride);
    }
    for (unsigned i = 0 ; i < scalar_arg_count ; i++) {
        double scalar = arr_scalar[i];
        status |= clSetKernelArg(
            kernel,
            i + 3 * vector_arg_count,
            sizeof(
            double),
            &scalar
        );
    }

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

static inline void gpu_matrix_vector_buffer_reduce_BANG_helper(
    vector_buffer * v_x,
    cl_command_queue cmd_queue,
    kernel_type_t kernel_id
){
    cl_event enqueue_events[1];
    cl_kernel kernel;
    cl_int status;
    index_t remaining_length;

    // Then, compute the sum 
    kernel = kernels_get(
        context_get(),
        device_get(),
        kernel_id 
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

static inline cl_mem gpu_matrix_vector_buffer_reduce_index_BANG_helper(
    vector_buffer * v_x,
    cl_command_queue cmd_queue,
    kernel_type_t kernel_id
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
        kernel_id 
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

extern void gpu_matrix_vector_buffer_axpy_BANG(
    vector_buffer * v_x,
    vector_buffer * v_y,
    double alpha,
    cl_command_queue cmd_queue
) {
    vector_buffer * arr_vector_buffer[2] = { v_x, v_y };
    gpu_matrix_vector_buffer_map_BANG_helper(
        2, arr_vector_buffer,
        1, &alpha,
        cmd_queue,
        KERNEL_VECTOR_AXPY_BANG
    );
}

extern void gpu_matrix_vector_buffer_asum_BANG(
    vector_buffer * v_x,
    cl_command_queue cmd_queue
) {
    gpu_matrix_vector_buffer_reduce_BANG_helper(
        v_x,
        cmd_queue,
        KERNEL_VECTOR_ASUM
    );
}

extern void gpu_matrix_vector_buffer_mul_BANG(
    vector_buffer * v_x,
    vector_buffer * v_y,
    cl_command_queue cmd_queue
) {
    vector_buffer * arr_vector_buffer[2] = { v_x, v_y };

    gpu_matrix_vector_buffer_map_BANG_helper(
        2, arr_vector_buffer,
        0, NULL,
        cmd_queue,
        KERNEL_VECTOR_MUL_BANG
    );
}

extern void gpu_matrix_vector_buffer_square_BANG(
    vector_buffer * v_x,
    cl_command_queue cmd_queue
) {
    gpu_matrix_vector_buffer_map_BANG_helper(
        1, &v_x,
        0, NULL,
        cmd_queue,
        KERNEL_VECTOR_SQUARE_BANG 
    );
}

void gpu_matrix_vector_buffer_rot_BANG(
    vector_buffer* v_x,
    vector_buffer* v_y,
    double c,
    double s,
    cl_command_queue cmd_queue
) {
    vector_buffer * vector_buffer_arr[2] = { v_x, v_y };
    double scalar_arr[2] = { c, s };
    gpu_matrix_vector_buffer_map_BANG_helper(
        2, vector_buffer_arr,
        2, scalar_arr,
        cmd_queue,
        KERNEL_VECTOR_ROT_BANG 
    );
}

extern void gpu_matrix_vector_buffer_abs_BANG(
    vector_buffer * v_x,
    cl_command_queue cmd_queue
)  {
    gpu_matrix_vector_buffer_map_BANG_helper(
        1, &v_x,
        0, NULL,
        cmd_queue,
        KERNEL_VECTOR_ABS_BANG
    );
}

extern void gpu_matrix_vector_buffer_max_BANG(
    vector_buffer* v_x,
    cl_command_queue cmd_queue
) {
    gpu_matrix_vector_buffer_reduce_BANG_helper(
        v_x,
        cmd_queue,
        KERNEL_VECTOR_MAX_BANG 
    );
}

extern void gpu_matrix_vector_buffer_min_BANG(
    vector_buffer * v_x,
    cl_command_queue cmd_queue
) {
    gpu_matrix_vector_buffer_reduce_BANG_helper(
        v_x,
        cmd_queue,
        KERNEL_VECTOR_MIN_BANG 
    );
}

extern cl_mem gpu_matrix_vector_buffer_imax(
    vector_buffer * v_x,
    cl_command_queue cmd_queue
) {
    return gpu_matrix_vector_buffer_reduce_index_BANG_helper(
        v_x,
        cmd_queue,
        KERNEL_VECTOR_IMAX_BANG
    );
}

extern cl_mem gpu_matrix_vector_buffer_imin(
    vector_buffer * v_x,
    cl_command_queue cmd_queue
) {
    return gpu_matrix_vector_buffer_reduce_index_BANG_helper(
        v_x,
        cmd_queue,
        KERNEL_VECTOR_IMIN_BANG
    );
}

