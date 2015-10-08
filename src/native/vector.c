// How to DRY this source code up, there is too many repetition

#include "vector.h"

#define BUFFERS_CREATE_AND_WRITE

const index_t ONE = 1;

void describe_event(const char * description, cl_ulong time) {
    printf("%s: %.3f milliseconds\n", description, ((double) time * 0.000001));
}

void gpu_matrix_release(vector * x) {
    // TODO, don't just free
}

vector_buffer gpu_matrix_vector_to_vector_buffer(vector * v, cl_mem buffer) {
    vector_buffer obj;
    obj.buffer = buffer;
    obj.length = v->length;
    obj.stride = v->stride;
    obj.datasize = sizeof(double) * v->length;
    return obj;
}

size_t gpu_matrix_vector_datasize(vector * x) {
    return x->length * sizeof(double);
}

vector * gpu_matrix_vector_copy(vector * v_x) {
    vector * copy = malloc(sizeof(vector));
    copy->data = malloc(v_x->length * sizeof(double));
    copy->length = v_x->length;
    copy->stride = 1;

    for (index_t i = 0 ; i < v_x->length ; i++) {
        copy->data[i] = v_x->data[i * v_x->stride];
    }

    return copy;
}

void gpu_matrix_vector_axpy(vector * v_x, double alpha, vector * v_y) {
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
    cl_mem buffer_x, buffer_y, buffer_output, buffer_local_cache;
    vector_buffer buffer_v_x, buffer_v_y;

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
    buffer_v_x = gpu_matrix_vector_to_vector_buffer(v_x, buffer_x);
    buffer_v_y = gpu_matrix_vector_to_vector_buffer(v_y, buffer_y);
    
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
    gpu_matrix_vector_buffer_axpy_BANG(
        &buffer_v_x,
        &buffer_v_y,
        alpha,
        cmd_queue
    );

    clEnqueueReadBuffer(
        cmd_queue,
        buffer_x,
        CL_TRUE,
        0,
        datasize,
        v_x->data,
        0,
        NULL,
        read_buffer_events
    );

    clReleaseEvent(write_buffer_events[0]);
    clReleaseEvent(write_buffer_events[1]);
    clReleaseEvent(read_buffer_events[0]);
}

void gpu_matrix_vector_scal(vector * v_x, double alpha) {
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
    cl_mem buffer_x, buffer_y, buffer_output, buffer_local_cache;
    vector_buffer buffer_v_x, buffer_v_y;

    buffer_x = buffers_create(
        CL_MEM_READ_ONLY,
        datasize,
        NULL,
        &status
    );
    buffer_v_x = gpu_matrix_vector_to_vector_buffer(v_x, buffer_x);
    
    status = clEnqueueWriteBuffer(
        cmd_queue, buffer_x,
        CL_FALSE, 0, datasize,
        v_x->data, 0, NULL, write_buffer_events
    );

    clWaitForEvents(1, write_buffer_events);
    gpu_matrix_vector_buffer_scal_BANG(
        &buffer_v_x,
        alpha,
        cmd_queue
    );

    clEnqueueReadBuffer(
        cmd_queue,
        buffer_x,
        CL_TRUE,
        0,
        datasize,
        v_x->data,
        0,
        NULL,
        read_buffer_events
    );

    clReleaseEvent(write_buffer_events[0]);
    clReleaseEvent(read_buffer_events[0]);
}

void gpu_matrix_vector_swap(vector * v_x, vector * v_y) {
    vector tmp_x = *v_x;
    *v_x = *v_y;
    *v_y = tmp_x;
}

void gpu_matrix_vector_rot(
    vector * v_x,
    vector * v_y,
    double c,
    double s
){
    cl_kernel kernel;
    cl_command_queue cmd_queue;
    cl_int status;
    cl_event read_buffer_events[2], write_buffer_events[2];
    size_t datasize = gpu_matrix_vector_datasize(v_x);
    cl_mem buffer_data_x, buffer_data_y;
    vector_buffer buffer_v_x, buffer_v_y;

    cmd_queue = clCreateCommandQueue(
        context_get(),
        device_get(),
        CL_QUEUE_PROFILING_ENABLE,
        &status
    );
    buffer_data_x = buffers_create(
        CL_MEM_READ_WRITE,
        datasize,
        NULL,
        &status
    );
    buffer_data_y = buffers_create(
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
    
    buffer_v_x = gpu_matrix_vector_to_vector_buffer(v_x, buffer_data_x);
    buffer_v_y = gpu_matrix_vector_to_vector_buffer(v_x, buffer_data_y);

    gpu_matrix_vector_buffer_rot_BANG(&buffer_v_x, &buffer_v_y, c, s, cmd_queue);

    status = clEnqueueReadBuffer(
        cmd_queue,
        buffer_v_x.buffer,
        CL_TRUE,
        0,
        datasize,
        v_x->data,
        0,
        NULL,
        read_buffer_events
    );
    status = clEnqueueReadBuffer(
        cmd_queue,
        buffer_v_y.buffer,
        CL_TRUE,
        0,
        datasize,
        v_y->data,
        0,
        NULL,
        read_buffer_events+1
    );

    clWaitForEvents(2, read_buffer_events);
}

double gpu_matrix_vector_dot(vector * v_x, vector * v_y) {
    size_t datasize = gpu_matrix_vector_datasize(v_x);
    size_t global_work_size[1];
    cl_int status;
    double output;
    cl_mem buffer_data_output, buffer_data_x, buffer_data_y;
    vector_buffer buffer_v_x, buffer_v_y;
    cl_command_queue cmd_queue;
    cl_event write_buffer_events[2],
             read_buffer_events[1];
  
    datasize = gpu_matrix_vector_datasize(v_x);
    global_work_size[0] = v_x->length;
    cmd_queue = clCreateCommandQueue(
        context_get(),
        device_get(),
        CL_QUEUE_PROFILING_ENABLE,
        &status
    );
    buffer_data_x = buffers_create(
        CL_MEM_READ_WRITE,
        datasize,
        NULL,
        &status
    );
    buffer_v_x = gpu_matrix_vector_to_vector_buffer(v_x, buffer_data_x);

    status = clEnqueueWriteBuffer(
        cmd_queue, buffer_data_x,
        CL_FALSE, 0, datasize,
        v_x->data, 0, NULL,
        write_buffer_events
    );
    buffer_data_y = buffers_create(
        CL_MEM_READ_ONLY,
        datasize,
        NULL,
        &status
    );
    status = clEnqueueWriteBuffer(
        cmd_queue, buffer_data_y,
        CL_FALSE, 0, datasize,
        v_y->data, 0, NULL,
        write_buffer_events+1
    );
    buffer_v_y = gpu_matrix_vector_to_vector_buffer(v_y, buffer_data_y);
    clWaitForEvents(2, write_buffer_events);

    gpu_matrix_vector_buffer_mul_BANG(&buffer_v_x, &buffer_v_y, cmd_queue); 
    gpu_matrix_vector_buffer_asum_BANG(&buffer_v_x, cmd_queue);

    clEnqueueReadBuffer(
        cmd_queue,
        buffer_v_x.buffer,
        CL_TRUE,
        0,
        sizeof(double),
        &output,
        0,
        NULL,
        read_buffer_events
    );
    clWaitForEvents(1, read_buffer_events);

    // Release buffer objects
    clReleaseMemObject(buffer_v_x.buffer);
    clReleaseEvent(read_buffer_events[0]);
    clReleaseEvent(write_buffer_events[0]);
    clReleaseEvent(write_buffer_events[1]);

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
    vector_buffer buffer_v_x;
    
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
    buffer_v_x = gpu_matrix_vector_to_vector_buffer(v_x, buffer_data);
    gpu_matrix_vector_buffer_asum_BANG(&buffer_v_x, cmd_queue);

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

    clReleaseMemObject(buffer_data);
    clReleaseEvent(read_buffer_events[0]);
    clReleaseEvent(write_buffer_events[0]);

    return output;
}

double gpu_matrix_vector_nrm2(vector * v_x) {
    size_t datasize = gpu_matrix_vector_datasize(v_x);
    cl_int status;
    double output;
    cl_mem buffer_data;
    cl_command_queue cmd_queue;
    cl_kernel kernel;
    index_t remaining_length = v_x->length;
    cl_event write_buffer_events[1],
             read_buffer_events[1];
    vector_buffer buffer_v_x;
    
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
    buffer_v_x = gpu_matrix_vector_to_vector_buffer(v_x, buffer_data);
    gpu_matrix_vector_buffer_square_BANG(&buffer_v_x, cmd_queue);
    gpu_matrix_vector_buffer_asum_BANG(&buffer_v_x, cmd_queue);

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

    clReleaseMemObject(buffer_data);
    clReleaseEvent(read_buffer_events[0]);
    clReleaseEvent(write_buffer_events[0]);

    return output;
}

double gpu_matrix_vector_amax(vector * v_x) {
    size_t datasize = gpu_matrix_vector_datasize(v_x);
    cl_int status;
    double output;
    cl_mem buffer_data;
    cl_command_queue cmd_queue;
    cl_kernel kernel;
    index_t remaining_length = v_x->length;
    cl_event write_buffer_events[1],
             read_buffer_events[1];
    vector_buffer buffer_v_x;
    
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
    buffer_v_x = gpu_matrix_vector_to_vector_buffer(v_x, buffer_data);
    gpu_matrix_vector_buffer_abs_BANG(&buffer_v_x, cmd_queue);
    gpu_matrix_vector_buffer_max_BANG(&buffer_v_x, cmd_queue);

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

    clReleaseMemObject(buffer_data);
    clReleaseEvent(read_buffer_events[0]);
    clReleaseEvent(write_buffer_events[0]);

    return output;
}

double gpu_matrix_vector_amin(vector * v_x) {
    size_t datasize = gpu_matrix_vector_datasize(v_x);
    cl_int status;
    double output;
    cl_mem buffer_data;
    cl_command_queue cmd_queue;
    cl_kernel kernel;
    index_t remaining_length = v_x->length;
    cl_event write_buffer_events[1],
             read_buffer_events[1];
    vector_buffer buffer_v_x;
    
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
    buffer_v_x = gpu_matrix_vector_to_vector_buffer(v_x, buffer_data);
    gpu_matrix_vector_buffer_abs_BANG(&buffer_v_x, cmd_queue);
    gpu_matrix_vector_buffer_min_BANG(&buffer_v_x, cmd_queue);

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

    clReleaseMemObject(buffer_data);
    clReleaseEvent(read_buffer_events[0]);
    clReleaseEvent(write_buffer_events[0]);

    return output;
}

index_t gpu_matrix_vector_imax(vector * v_x) {
    size_t datasize = gpu_matrix_vector_datasize(v_x);
    cl_int status;
    index_t output;
    cl_mem buffer_data, buffer_indices;
    cl_command_queue cmd_queue;
    cl_kernel kernel;
    index_t remaining_length = v_x->length;
    cl_event write_buffer_events[1],
             read_buffer_events[1];
    vector_buffer buffer_v_x;
    
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
    buffer_v_x = gpu_matrix_vector_to_vector_buffer(v_x, buffer_data);
    buffer_indices = gpu_matrix_vector_buffer_imax(&buffer_v_x, cmd_queue);

    // Print write buffer profilling information
    clEnqueueReadBuffer(
        cmd_queue,
        buffer_indices,
        CL_TRUE,
        0,
        sizeof(index_t),
        &output,
        0,
        NULL,
        read_buffer_events
    );

    clReleaseMemObject(buffer_data);
    clReleaseMemObject(buffer_indices);
    clReleaseEvent(read_buffer_events[0]);
    clReleaseEvent(write_buffer_events[0]);

    return output;
}

index_t gpu_matrix_vector_imin(vector * v_x) {
    size_t datasize = gpu_matrix_vector_datasize(v_x);
    cl_int status;
    index_t output;
    cl_mem buffer_data, buffer_indices;
    cl_command_queue cmd_queue;
    cl_kernel kernel;
    index_t remaining_length = v_x->length;
    cl_event write_buffer_events[1],
             read_buffer_events[1];
    vector_buffer buffer_v_x;
    
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
    buffer_v_x = gpu_matrix_vector_to_vector_buffer(v_x, buffer_data);
    buffer_indices = gpu_matrix_vector_buffer_imin(&buffer_v_x, cmd_queue);

    // Print write buffer profilling information
    status = clEnqueueReadBuffer(
        cmd_queue,
        buffer_indices,
        CL_TRUE,
        0,
        sizeof(index_t),
        &output,
        0,
        NULL,
        read_buffer_events
    );

    clReleaseMemObject(buffer_data);
    clReleaseMemObject(buffer_indices);
    clReleaseEvent(read_buffer_events[0]);
    clReleaseEvent(write_buffer_events[0]);

    return output;
}

index_t gpu_matrix_vector_iamax(vector * v_x) {
    size_t datasize = gpu_matrix_vector_datasize(v_x);
    cl_int status;
    index_t output;
    cl_mem buffer_data, buffer_indices;
    cl_command_queue cmd_queue;
    cl_kernel kernel;
    index_t remaining_length = v_x->length;
    cl_event write_buffer_events[1],
             read_buffer_events[1];
    vector_buffer buffer_v_x;
    
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
    buffer_v_x = gpu_matrix_vector_to_vector_buffer(v_x, buffer_data);
    gpu_matrix_vector_buffer_abs_BANG(&buffer_v_x, cmd_queue);
    buffer_indices = gpu_matrix_vector_buffer_imax(&buffer_v_x, cmd_queue);

    // Print write buffer profilling information
    status = clEnqueueReadBuffer(
        cmd_queue,
        buffer_indices,
        CL_TRUE,
        0,
        sizeof(index_t),
        &output,
        0,
        NULL,
        read_buffer_events
    );

    clReleaseMemObject(buffer_data);
    clReleaseMemObject(buffer_indices);
    clReleaseEvent(read_buffer_events[0]);
    clReleaseEvent(write_buffer_events[0]);

    return output;
}

index_t gpu_matrix_vector_iamin(vector * v_x) {
    size_t datasize = gpu_matrix_vector_datasize(v_x);
    cl_int status;
    index_t output;
    cl_mem buffer_data, buffer_indices;
    cl_command_queue cmd_queue;
    cl_kernel kernel;
    index_t remaining_length = v_x->length;
    cl_event write_buffer_events[1],
             read_buffer_events[1];
    vector_buffer buffer_v_x;
    
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

    buffer_v_x = gpu_matrix_vector_to_vector_buffer(v_x, buffer_data);
    gpu_matrix_vector_buffer_abs_BANG(&buffer_v_x, cmd_queue);
    buffer_indices = gpu_matrix_vector_buffer_imin(&buffer_v_x, cmd_queue);

    // Print write buffer profilling information
    status = clEnqueueReadBuffer(
        cmd_queue,
        buffer_indices,
        CL_TRUE,
        0,
        sizeof(index_t),
        &output,
        0,
        NULL,
        read_buffer_events
    );

    clReleaseMemObject(buffer_data);
    clReleaseMemObject(buffer_indices);
    clReleaseEvent(read_buffer_events[0]);
    clReleaseEvent(write_buffer_events[0]);

    return output;
}

// Arimethic stuff
vector * gpu_matrix_vector_add_scalar(vector * v_x, double alpha) {
    size_t datasize;
    cl_int status;
    vector * output;
    cl_mem buffer_data_x;
    vector_buffer buffer_v_x;
    cl_command_queue cmd_queue;
    cl_kernel kernel;
    cl_event write_buffer_events[1],
             read_buffer_events[1];

    output = malloc(sizeof(vector));
    output->length = v_x->length;
    output->stride = 1;
    output->data = malloc(sizeof(double) * v_x->length);

    datasize = gpu_matrix_vector_datasize(v_x);
    cmd_queue = clCreateCommandQueue(
        context_get(),
        device_get(),
        CL_QUEUE_PROFILING_ENABLE,
        &status
    );
    buffer_data_x = buffers_create(
        CL_MEM_READ_WRITE,
        datasize,
        NULL,
        &status
    );
    buffer_v_x = gpu_matrix_vector_to_vector_buffer(
        v_x, buffer_data_x
    );

    status = clEnqueueWriteBuffer(
        cmd_queue, buffer_data_x,
        CL_TRUE, 0, datasize,
        v_x->data, 0, NULL, write_buffer_events
    );
    clWaitForEvents(1, write_buffer_events);

    gpu_matrix_vector_buffer_add_scalar_BANG(&buffer_v_x, alpha, cmd_queue);

    // Print write buffer profilling information
    status = clEnqueueReadBuffer(
        cmd_queue,
        buffer_data_x,
        CL_TRUE,
        0,
        datasize,
        output->data,
        0,
        NULL,
        read_buffer_events
    );
    clReleaseMemObject(buffer_data_x);
    clReleaseEvent(read_buffer_events[0]);
    clReleaseEvent(write_buffer_events[0]);

    return output;
}

vector * gpu_matrix_vector_add_2(vector * v_x, vector * v_y) {
    size_t datasize;
    cl_int status;
    vector * output;
    cl_mem buffer_data_x, buffer_data_y;
    vector_buffer buffer_v_x, buffer_v_y;
    cl_command_queue cmd_queue;
    cl_kernel kernel;
    cl_event write_buffer_events[2],
             read_buffer_events[1];

    output = malloc(sizeof(vector));
    output->length = v_x->length;
    output->stride = 1;
    output->data = malloc(sizeof(double) * v_x->length);

    datasize = gpu_matrix_vector_datasize(v_x);
    cmd_queue = clCreateCommandQueue(
        context_get(),
        device_get(),
        CL_QUEUE_PROFILING_ENABLE,
        &status
    );
    buffer_data_x = buffers_create(
        CL_MEM_READ_WRITE,
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
    buffer_v_x = gpu_matrix_vector_to_vector_buffer(
        v_x, buffer_data_x
    );
    buffer_v_y = gpu_matrix_vector_to_vector_buffer(
        v_y, buffer_data_y
    );

    status = clEnqueueWriteBuffer(
        cmd_queue, buffer_data_x,
        CL_TRUE, 0, datasize,
        v_x->data, 0, NULL, write_buffer_events
    );
    status = clEnqueueWriteBuffer(
        cmd_queue, buffer_data_y,
        CL_TRUE, 0, datasize,
        v_y->data, 0, NULL, write_buffer_events+1
    );
    clWaitForEvents(2, write_buffer_events);

    gpu_matrix_vector_buffer_add_BANG(&buffer_v_x, &buffer_v_y, cmd_queue);

    // Print write buffer profilling information
    status = clEnqueueReadBuffer(
        cmd_queue,
        buffer_data_x,
        CL_TRUE,
        0,
        datasize,
        output->data,
        0,
        NULL,
        read_buffer_events
    );
    clReleaseMemObject(buffer_data_x);
    clReleaseMemObject(buffer_data_y);
    clReleaseEvent(read_buffer_events[0]);
    clReleaseEvent(write_buffer_events[0]);
    clReleaseEvent(write_buffer_events[1]);

    return output;
}

vector * gpu_matrix_vector_add_arbitary(unsigned count, vector * arr_v[]) {
    if (count == 1) {
        return gpu_matrix_vector_copy(arr_v[0]);
    } else if (count == 0) {
        // Have fun mingling with the NULL pointer
        return NULL;
    } 

    size_t datasize;
    cl_int status;
    vector * output;
    cl_mem buffer_data;
    cl_command_queue cmd_queue;
    cl_kernel kernel;
    cl_event write_buffer_events_arr[10];
    cl_event *write_buffer_events,
             read_buffer_events[1];
    vector_buffer buffer_v_this;

    if (count < 10) {
        write_buffer_events = write_buffer_events_arr;
    } else {
        write_buffer_events = malloc(sizeof(cl_event) * count);
    }
    // increate write_buffer_events and read_buffer_events if count is not enough

    output = malloc(sizeof(vector));
    output->length = arr_v[0]->length;
    output->stride = 1;
    output->data = malloc(sizeof(double) * arr_v[0]->length);
    datasize = gpu_matrix_vector_datasize(arr_v[0]);
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
    buffer_v_this = gpu_matrix_vector_to_vector_buffer(
        arr_v[0], buffer_data
    );
    status = clEnqueueWriteBuffer(
        cmd_queue, buffer_data,
        CL_TRUE, 0, datasize,
        arr_v[0]->data, 0, NULL, write_buffer_events
    );
    clWaitForEvents(1, write_buffer_events);
    clReleaseEvent(write_buffer_events[0]);

    // Naive implementation, Improvements later
    for (unsigned i = 1 ; i < count ; i++) {
        cl_mem buffer_data_next;
        vector_buffer buffer_v_next;

        buffer_data_next = buffers_create(
            CL_MEM_READ_ONLY,
            datasize,
            NULL,
            &status
        );
        status = clEnqueueWriteBuffer(
            cmd_queue, buffer_data_next,
            CL_TRUE, 0 ,datasize,
            arr_v[i]->data, 0, NULL, write_buffer_events+i
        );
        buffer_v_next = gpu_matrix_vector_to_vector_buffer(
            arr_v[i], buffer_data_next 
        );
        clWaitForEvents(1, write_buffer_events+i);
        gpu_matrix_vector_buffer_add_BANG(&buffer_v_this, &buffer_v_next, cmd_queue);

        clReleaseEvent(write_buffer_events[i]);
        clReleaseMemObject(buffer_data_next);
    }

    // Print write buffer profilling information
    status = clEnqueueReadBuffer(
        cmd_queue,
        buffer_data,
        CL_TRUE,
        0,
        datasize,
        output->data,
        0,
        NULL,
        read_buffer_events 
    );
    clReleaseMemObject(buffer_data);
    clReleaseEvent(read_buffer_events[0]);

    if (count >= 10) {
        free(write_buffer_events); 
    }

    return output;
}

