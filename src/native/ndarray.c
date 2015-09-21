#include <CL/cl.h>
#include <stdbool.h>
#include <stdio.h>

#include "ndarray.h"
#include "utils.h"

// Force the context and device to intialize
cl_mem map_helper(void * ptr_buffer_x,
                  cl_int size_buffer_x,
                  void * ptr_buffer_y,
                  cl_int size_buffer_y,
                  unsigned number_of_elements,
                  size_t datasize,
                  cl_kernel kernel) {

    cl_int status;
    cl_mem buffer_output;
    cl_command_queue cmd_queue;
    size_t global_work_size[1] = { number_of_elements };

    cmd_queue = command_queue_create(0, &status);           
    buffer_output = buffers_create(CL_MEM_WRITE_ONLY,       
            datasize, NULL, &status);                       
    status = clSetKernelArg(kernel, 0, size_buffer_x,      
            ptr_buffer_x);                             
    status |= clSetKernelArg(kernel, 1, size_buffer_y, 
            ptr_buffer_y);                             
    status |= clSetKernelArg(kernel, 2, sizeof(cl_mem),     
            (void*) &buffer_output);                        
    status = clEnqueueNDRangeKernel(cmd_queue, kernel, 1,   
            NULL, global_work_size, NULL, 0, NULL, NULL);   
    clReleaseCommandQueue(cmd_queue);

    return buffer_output;
}

void map_bang_helper(void * ptr_buffer_x,
                     cl_int size_buffer_x,
                     void * ptr_buffer_y,
                     cl_int size_buffer_y,
                     int number_of_elements,
                     size_t datasize,
                     cl_kernel kernel) {

    cl_int status;
    cl_command_queue cmd_queue;
    size_t global_work_size[1] = { number_of_elements };

    cmd_queue = command_queue_create(0, &status);           
    status = clSetKernelArg(kernel, 0, size_buffer_x,      
            ptr_buffer_x);                             
    status |= clSetKernelArg(kernel, 1, size_buffer_y, 
            ptr_buffer_y);                             
    
    status = clEnqueueNDRangeKernel(cmd_queue, kernel, 1,   
            NULL, global_work_size, NULL, 0, NULL, NULL);   
    clReleaseCommandQueue(cmd_queue);
}

void map_bang_factory(ndarray * arr_x, const ndarray * arr_y, kernel_type_t kernel_id) {
    const unsigned datasize = ndarray_datasize(arr_x);
    const unsigned number_of_elements = ndarray_elements_count(arr_x);
    cl_kernel kernel;
    cl_mem buffer_x, buffer_y;
    cl_int status;
    cl_command_queue cmd_queue;

    cmd_queue = clCreateCommandQueue(context_get(), device_get(), 0, &status);
    buffer_x = buffers_create(CL_MEM_READ_WRITE, datasize,   
            NULL, &status);                                 
    buffer_y = buffers_create(CL_MEM_READ_ONLY, datasize,   
            NULL, &status);                     
    status = clEnqueueWriteBuffer(cmd_queue, buffer_x,      
            CL_FALSE, 0, datasize,                          
            arr_x->data, 0, NULL, NULL);                    
    status = clEnqueueWriteBuffer(cmd_queue, buffer_y,      
            CL_FALSE, 0, datasize,                          
            arr_y->data, 0, NULL, NULL);
    kernel = kernels_get(context_get(), device_get(), kernel_id);
    map_bang_helper(
        &buffer_x, sizeof(cl_mem),
        &buffer_y, sizeof(cl_mem),
        number_of_elements,
        datasize,
        kernel
    );
    clEnqueueReadBuffer(cmd_queue, buffer_x, CL_TRUE, 0,
            datasize, arr_x->data, 0, NULL, NULL);

    clReleaseMemObject(buffer_x);
    clReleaseMemObject(buffer_y);
    clReleaseCommandQueue(cmd_queue);
}

void map_scalar_bang_factory(ndarray * arr_x, double y, kernel_type_t kernel_id) {
    const unsigned datasize = ndarray_datasize(arr_x);
    const unsigned number_of_elements = ndarray_elements_count(arr_x);
    cl_kernel kernel;
    cl_mem buffer_x;
    cl_int status;
    cl_command_queue cmd_queue;

    cmd_queue = clCreateCommandQueue(context_get(), device_get(), 0, &status);
    buffer_x = buffers_create(CL_MEM_READ_WRITE, datasize,   
            NULL, &status);                                 
    status = clEnqueueWriteBuffer(cmd_queue, buffer_x,      
            CL_FALSE, 0, datasize,                          
            arr_x->data, 0, NULL, NULL);                    
    kernel = kernels_get(context_get(), device_get(), kernel_id);
    map_bang_helper(
        &buffer_x, sizeof(cl_mem),
        &y, sizeof(double),
        number_of_elements,
        datasize,
        kernel
    );
    clEnqueueReadBuffer(cmd_queue, buffer_x, CL_TRUE, 0,
            datasize, arr_x->data, 0, NULL, NULL);

    clReleaseMemObject(buffer_x);
    clReleaseCommandQueue(cmd_queue);
}

ndarray * map_scalar_factory(const ndarray * arr_x, double y, kernel_type_t kernel_id) {
    const unsigned datasize = ndarray_datasize(arr_x);
    ndarray * output = ndarray_clone_structure(arr_x);
    cl_kernel kernel;
    cl_int status;
    cl_command_queue cmd_queue;
    cl_mem buffer_output, buffer_x, buffer_y;
    unsigned number_of_elements;

    // TODO: Stride resizing
    number_of_elements = ndarray_elements_count(arr_x);
    cmd_queue = clCreateCommandQueue(context_get(), device_get(), 0, &status);
    buffer_x = buffers_create(CL_MEM_READ_ONLY, datasize,   
            NULL, &status);                                 
    status = clEnqueueWriteBuffer(cmd_queue, buffer_x,      
            CL_FALSE, 0, datasize,                          
            arr_x->data, 0, NULL, NULL);                    
    kernel = kernels_get(context_get(), device_get(), kernel_id);
    buffer_output = map_helper(
        &buffer_x, sizeof(cl_mem),
        &y, sizeof(double),
        number_of_elements,
        datasize,
        kernel    
    );
    clEnqueueReadBuffer(cmd_queue, buffer_output, CL_TRUE,
            0, datasize, output->data, 0, NULL, NULL);

    clReleaseMemObject(buffer_x);
    clReleaseMemObject(buffer_y);
    clReleaseMemObject(buffer_output);
    clReleaseCommandQueue(cmd_queue);

    return output;
}

ndarray * map_factory(const ndarray * arr_x, const ndarray * arr_y, kernel_type_t kernel_id) {
    const unsigned datasize = ndarray_datasize(arr_x);
    ndarray * output = ndarray_clone_structure(arr_x);
    cl_kernel kernel;
    cl_int status;
    cl_command_queue cmd_queue;
    cl_mem buffer_output, buffer_x, buffer_y, buffer_shape_x, buffer_strides_x, buffer_shape_y, buffer_strides_y;
    unsigned number_of_elements = ndarray_elements_count(arr_x);
    unsigned ndims = arr_x->ndims;
    size_t * global_work_size = (size_t*) ((arr_x->ndims >= arr_y->ndims) ? arr_x->shape : arr_y->shape);
    number_of_elements = ndarray_elements_count(arr_x);
    cmd_queue = clCreateCommandQueue(context_get(), device_get(), 0, &status);
    buffer_x = buffers_create(CL_MEM_READ_ONLY, datasize,   
            NULL, &status);                                 
    buffer_y = buffers_create(CL_MEM_READ_ONLY, datasize,   
            NULL, &status);                                 
    buffer_shape_x = buffers_create(CL_MEM_READ_ONLY, sizeof(index_t) * ndims, NULL, &status);
    buffer_strides_x = buffers_create(CL_MEM_READ_ONLY, sizeof(index_t) * ndims, NULL, &status);
    buffer_shape_y = buffers_create(CL_MEM_READ_ONLY, sizeof(index_t) * ndims, NULL, &status);
    buffer_strides_y = buffers_create(CL_MEM_READ_ONLY, sizeof(index_t) * ndims, NULL, &status);
    buffer_output = buffers_create(CL_MEM_WRITE_ONLY,       
            datasize, NULL, &status);                       
    status = clEnqueueWriteBuffer(cmd_queue, buffer_x,      
            CL_FALSE, 0, datasize,                          
            arr_x->data, 0, NULL, NULL);
    status = clEnqueueWriteBuffer(cmd_queue, buffer_shape_x,      
            CL_FALSE, 0, sizeof(index_t) * arr_x->ndims,                          
            arr_x->shape, 0, NULL, NULL);
    status = clEnqueueWriteBuffer(cmd_queue, buffer_strides_x,      
            CL_FALSE, 0, sizeof(index_t) * arr_x->ndims,                          
            arr_x->strides, 0, NULL, NULL);
    status = clEnqueueWriteBuffer(cmd_queue, buffer_y,      
            CL_FALSE, 0, datasize,                          
            arr_y->data, 0, NULL, NULL);
    status = clEnqueueWriteBuffer(cmd_queue, buffer_shape_y,
            CL_FALSE, 0, sizeof(index_t) * arr_y->ndims,                          
            arr_y->shape, 0, NULL, NULL);
    status = clEnqueueWriteBuffer(cmd_queue, buffer_strides_y,      
            CL_FALSE, 0, sizeof(index_t) * 2,
            arr_y->strides, 0, NULL, NULL);
    kernel = kernels_get(context_get(), device_get(), kernel_id);
    cmd_queue = command_queue_create(0, &status);           
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem),      
            &buffer_x);                             
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem),
            &buffer_shape_x);
    status = clSetKernelArg(kernel, 2, sizeof(cl_mem),
            &buffer_strides_x);
    status = clSetKernelArg(kernel, 3, sizeof(unsigned long long),
            &ndims);
    status = clSetKernelArg(kernel, 4, sizeof(cl_mem),      
            &buffer_y);                             
    status = clSetKernelArg(kernel, 5, sizeof(cl_mem),
            &buffer_shape_y);
    status = clSetKernelArg(kernel, 6, sizeof(cl_mem),
            &buffer_strides_y);
    status = clSetKernelArg(kernel, 7, sizeof(unsigned long long),
            &ndims);
    status = clSetKernelArg(kernel, 8, sizeof(cl_mem),
            &buffer_output);

    status = clEnqueueNDRangeKernel(cmd_queue, kernel, 2,
            NULL, global_work_size, NULL, 0, NULL, NULL);   
    clEnqueueReadBuffer(cmd_queue, buffer_output, CL_TRUE, 0, datasize, output->data, 0, NULL, NULL);

    clReleaseMemObject(buffer_x);
    clReleaseMemObject(buffer_y);
    clReleaseMemObject(buffer_output);
    clReleaseCommandQueue(cmd_queue);

    return output;
}

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

index_t * ndarray_make_basic_strides(index_t ndims, index_t * shape) {
    index_t * strides = malloc(sizeof(index_t) * ndims);
    strides[ndims - 1] = 1;
    for (int i = ndims - 2 ; i >= 0 ; i--) {
        strides[i] = strides[i+1] * shape[i+1];
    }

    return strides;
}

ndarray * ndarray_constructor(double * data, index_t ndims, index_t * shape, index_t * strides) {
    ndarray tmp;
    tmp.data = data;
    tmp.shape = shape;
    tmp.ndims = ndims;
    tmp.strides = strides;
    return ndarray_clone(&tmp);
}

ndarray * ndarray_constructor_from_shape(index_t ndims, index_t * shape) {
    size_t datasize;
    index_t number_of_elements = 1;
    for (int i = 0 ; i < ndims ; i++) {
        number_of_elements *= shape[i];
    }
    datasize = number_of_elements * sizeof(size_t);
    // datasize immutable from this point!

    ndarray * output = malloc(sizeof(ndarray));
    output->data     = (double*) malloc(datasize);
    output->ndims    = ndims;
    output->shape    = array_index_t_copy(shape, ndims);
    output->strides  = ndarray_make_basic_strides(ndims, shape);

    for(index_t i = 0 ; i < number_of_elements ; i++) {
        output->data[i] = 0.0;
    }

    return output;
}

ndarray * ndarray_clone_structure(const ndarray * arr_x) {
    const unsigned datasize = ndarray_datasize(arr_x);
    ndarray * output = malloc(sizeof(ndarray));
    
    output->data     = (double*) malloc(datasize);
    output->strides  = array_index_t_copy(arr_x->strides, (unsigned long long) arr_x->ndims);
    output->shape    = array_index_t_copy(arr_x->shape, arr_x->ndims);
    output->ndims    = arr_x->ndims;


    return output;
}

ndarray * ndarray_clone(const ndarray * arr_x) {
    ndarray * output = ndarray_clone_structure(arr_x);
    for (int i = 0 ; i < ndarray_elements_count(arr_x) ; i++) {
        output->data[i] = arr_x->data[i];
    }

    return output;
}

// getter and setters

void ndarray_set_1d(ndarray * arr, long i, double v) {
    arr->data[i] = v;
}

void ndarray_set_2d(ndarray * arr, long i, long j, double v) {
    arr->data[i * arr->strides[0] + j * arr->strides[1]] = v;
}

void ndarray_set_nd(ndarray * arr, const long * indexes, double v) {
    long idx = 0;
    for (long i = 0 ; i < arr->ndims ; i++) {
        idx += arr->strides[i] * indexes[i];
    }
    arr->data[idx] = v;
}

// Arimethic ops

ndarray * ndarray_add(const ndarray * arr_x, const ndarray * arr_y) {
    return map_factory(arr_x, arr_y, KERNEL_ADD);
}

ndarray * ndarray_add_scalar(const ndarray * arr_x, const double y) {
    return map_scalar_factory(arr_x, y, KERNEL_ADD_SCALAR);
}

void ndarray_addbang(ndarray * arr_x, const ndarray * arr_y) {
    map_bang_factory(arr_x, arr_y, KERNEL_ADD_BANG);
}

void ndarray_add_scalar_bang(ndarray * arr_x, const double y) {
    map_scalar_bang_factory(arr_x, y, KERNEL_ADD_SCALAR_BANG);
}
 
ndarray * ndarray_sub(const ndarray * arr_x, const ndarray * arr_y) {
    return map_factory(arr_x, arr_y, KERNEL_SUB);
}

ndarray * ndarray_sub_scalar(const ndarray * arr_x, const double y) {
    return map_scalar_factory(arr_x, y, KERNEL_SUB_SCALAR);
}

void ndarray_subbang(ndarray * arr_x, const ndarray * arr_y) {
    map_bang_factory(arr_x, arr_y, KERNEL_SUB_BANG);
}

void ndarray_sub_scalar_bang(ndarray * arr_x, const double y) {
    map_scalar_bang_factory(arr_x, y, KERNEL_SUB_SCALAR_BANG);
}
 
ndarray * ndarray_mul(const ndarray * arr_x, const ndarray * arr_y) {
    return map_factory(arr_x, arr_y, KERNEL_MUL);
}

ndarray * ndarray_mul_scalar(const ndarray * arr_x, const double y) {
    return map_scalar_factory(arr_x, y, KERNEL_MUL_SCALAR);
}

void ndarray_mulbang(ndarray * arr_x, const ndarray * arr_y) {
    map_bang_factory(arr_x, arr_y, KERNEL_MUL_BANG);
}

void ndarray_mul_scalar_bang(ndarray * arr_x, const double y) {
    map_scalar_bang_factory(arr_x, y, KERNEL_MUL_SCALAR_BANG);
}
 
ndarray * ndarray_div(const ndarray * arr_x, const ndarray * arr_y) {
    return map_factory(arr_x, arr_y, KERNEL_DIV);
}

ndarray * ndarray_div_scalar(const ndarray * arr_x, const double y) {
    return map_scalar_factory(arr_x, y, KERNEL_DIV_SCALAR);
}

void ndarray_divbang(ndarray * arr_x, const ndarray * arr_y) {
    map_bang_factory(arr_x, arr_y, KERNEL_DIV_BANG);
}

void ndarray_div_scalar_bang(ndarray * arr_x, const double y) {
    map_scalar_bang_factory(arr_x, y, KERNEL_DIV_SCALAR_BANG);
}
