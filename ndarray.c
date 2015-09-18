#include <CL/cl.h>
#include <stdbool.h>
#include <stdio.h>

#include "ndarray.h"
#include "utils.h"

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

    return buffer_output;
}
/**
void map_bang_helper(void * buffer_x,
                     cl_int size_buffer_x,
                     void * buffer_y,
                     cl_int size_buffer_y,
                     int number_of_elements,
                     kernel_type_t kernel_id) {

}

void * map_bang_factory(const ndarray * arr_x, const ndarray * arr_y, kernel_type_t kernel_id) {
}

ndarray * map_scalar_factory(ndarray * arr_x, double x, kernel_type_t kernel_id) {
    
}
*/

ndarray * map_factory(const ndarray * arr_x, const ndarray * arr_y, kernel_type_t kernel_id) {
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
    buffer_y = buffers_create(CL_MEM_READ_ONLY, datasize,   
            NULL, &status);                                 
    status = clEnqueueWriteBuffer(cmd_queue, buffer_x,      
            CL_FALSE, 0, datasize,                          
            arr_x->data, 0, NULL, NULL);                    
    status = clEnqueueWriteBuffer(cmd_queue, buffer_y,      
            CL_FALSE, 0, datasize,                          
            arr_y->data, 0, NULL, NULL);
    kernel = kernels_get(context_get(), device_get(), kernel_id);
    buffer_output = map_helper(
        &buffer_x, sizeof(cl_mem),
        &buffer_y, sizeof(cl_mem),
        number_of_elements,
        datasize,
        kernel    
    );

    clEnqueueReadBuffer(cmd_queue, buffer_output, CL_TRUE,
            0, datasize, output->data, 0, NULL, NULL);

    return output;
}

ndarray * ndarray_add(const ndarray * arr_x, const ndarray * arr_y) {
    return map_factory(arr_x, arr_y, KERNEL_ADD);
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

ndarray * ndarray_clone_structure(const ndarray * arr_x) {
    const unsigned datasize = ndarray_datasize(arr_x);
    ndarray * output = malloc(sizeof(ndarray));
    
    output->data     = (double*) malloc(datasize);
    output->strides  = array_size_t_copy(arr_x->strides, (unsigned long long) arr_x->ndims);
    output->shape    = array_size_t_copy(arr_x->shape, arr_x->ndims);
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

// basic ops
ndarray * ndarray_add_scalar(const ndarray * arr_x, double y) {
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
    kernel = kernels_get(context_get(), device_get(), KERNEL_ADD_SCALAR);
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
/*
ndarray * ndarray_add(const ndarray * arr_x, const ndarray * arr_y) {
    const size_t * shape = arr_x->shape;
    const unsigned datasize = ndarray_datasize(arr_x);
    cl_kernel kernel;
    cl_int status;
    cl_mem buffer_x, buffer_output, buffer_y;
    cl_command_queue cmd_queue;
    ndarray * output = ndarray_clone_structure(arr_x);
    size_t global_work_size[1] = { ndarray_elements_count(arr_x) };

    cmd_queue = command_queue_create(0, &status);
    kernel = kernels_get(context_get(), device_get(), "add");
    buffer_x = buffers_create(CL_MEM_READ_ONLY, datasize, NULL, &status);
    buffer_y = buffers_create(CL_MEM_READ_ONLY, datasize, NULL, &status);
    buffer_output = buffers_create(CL_MEM_WRITE_ONLY, datasize, NULL, &status);
    status = clEnqueueWriteBuffer(cmd_queue, buffer_x, CL_FALSE, 0,
            datasize, arr_x->data, 0, NULL, NULL);
    status = clEnqueueWriteBuffer(cmd_queue, buffer_y, CL_FALSE, 0, 
            datasize, arr_y->data, 0, NULL, NULL);
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*) &buffer_x);
    status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*) &buffer_y);
    status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*) &buffer_output);
    status = clEnqueueNDRangeKernel(cmd_queue, kernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL);
    
    clEnqueueReadBuffer(cmd_queue, buffer_output, CL_TRUE, 0, ndarray_datasize(output), output->data, 0, NULL, NULL);

    return output;
}
*/
void ndarray_add_bang(ndarray * arr_x, const ndarray * arr_y) {
    const size_t * shape = arr_x->shape;
    const unsigned datasize = ndarray_datasize(arr_x);
    cl_kernel kernel;
    cl_int status;
    cl_mem buffer_x, buffer_y;
    cl_command_queue cmd_queue;
    size_t global_work_size[1] = { ndarray_elements_count(arr_x) };

    cmd_queue = command_queue_create(0, &status);
    kernel = kernels_get(context_get(), device_get(), KERNEL_ADD_BANG);
    buffer_x = buffers_create(CL_MEM_READ_WRITE, datasize, NULL, &status);
    buffer_y = buffers_create(CL_MEM_READ_ONLY, datasize, NULL, &status);
    status = clEnqueueWriteBuffer(cmd_queue, buffer_x, CL_FALSE, 0,
            datasize, arr_x->data, 0, NULL, NULL);
    status = clEnqueueWriteBuffer(cmd_queue, buffer_y, CL_FALSE, 0, 
            datasize, arr_y->data, 0, NULL, NULL);
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*) &buffer_x);
    status |= clSetKernelArg(kernel, 1, sizeof(double), (void*) &buffer_y);
    status = clEnqueueNDRangeKernel(cmd_queue, kernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL);
    
    clEnqueueReadBuffer(cmd_queue, buffer_x, CL_TRUE, 0,
            datasize, arr_x->data, 0, NULL, NULL);
}
