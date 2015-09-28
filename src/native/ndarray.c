// code structure for arimethic ops:

// NDArray:
// map_factory =>  takes in the numeric objects and kernel_id. Retrieve the appropriate kernel object for kernel_id and create a command queue. Array objects, kernel and cmd_queue are passed to map_helper. 
// map_helper  => rearranges, broadcast the objects as appropriate, and pass arguments as they are to the map_run_kernel and return the results as a ndarray object
// map_run_kernel => Takes in the objects and runs the kernels. modifieds that double*ret argument
// 
// map_factory => map_helper => map_run_kernel
// map_bang_factory => map_bang_helper => map_run_kernel
//
// Scalar:
// map_scalar_factory => map_scalar_run_kernel 
// map_saclar_bang_factory => map_scalar_run_kernel
// similiar to non-scalar counterparts
// note there is no helper for scalar as there is no position switching or coercingd
// required

#include <stdbool.h>
#include <stdio.h>

#include "ndarray.h"
#include "utils.h"

#define MAX(a,b) \
({ __typeof__ (a) _a = (a); \
   __typeof__ (b) _b = (b); \
 _a > _b ? _a : _b; })

// note: TRY_AND_CATCH_ERROR macro is defined everytime it is required - 
// we may require a slightly different version for depending on where it is used
// (eg: sometimes we want to return, sometimes we just want to exit)
// behavious is largely similiar in the same file, but should vary in different ones
#define TRY_AND_CATCH_ERROR(statement, status_var) \
statement;\
if (status_var != CL_SUCCESS) { \
    fprintf(stderr, "An error occured in running gpu.matrix at line %u of %s\n" , \
            __LINE__, __FILE__);  \
    exit(1); \
}

size_t * get_global_work_size(const ndarray * arr_x, const ndarray * arr_y, size_t * len) {
    const ndarray * big = (arr_x->ndims >= arr_y-> ndims) ? arr_x : arr_y;
    size_t ndims = (size_t) big->ndims;
    size_t * ret;

    if (ndims == 0) {
        *len = 1;
        ret = malloc(sizeof(size_t) * 1);
        ret[0] = 1;
        return ret;
    } else if (ndims > 0 && ndims <= 3)  {
        *len = ndims;
        ret = malloc(sizeof(size_t) * ndims);
        for (int i = 0 ; i < ndims ; i++) {
            ret[i] = (size_t) arr_x->shape[i]; 
        }
        return ret;
    } else {
        *len = 1;
        ret = malloc(sizeof(size_t));
        ret[0] = ndarray_elements_count(big); 
        return ret;
    }

    return ret;
}

// returns a new ndarray with the strides modified to be equal to those in the arguments
// assumes that both source and dest have similiar shapes
void coerce_stride_recur(
        const index_t dim,
        const ndarray * src,
        const index_t src_index_accum,
        ndarray * dest,
        const index_t dest_index_accum
    ) {
    // dim is the current dimension
    // source is the source ndarray object, dest is the counterpat
    // source_index and dest_index are just index accumulators
   
    for (index_t i = 0 ; i < src->shape[dim] ; i++) {
        const index_t dest_index = dest_index_accum + i * dest->strides[dim],
                      src_index = src_index_accum + i * src->strides[dim];
        if (dim == src->ndims - 1) {
            dest->data[dest_index] = src->data[src_index];
        } else {
            coerce_stride_recur(
                dim + 1,
                src,
                src_index,
                dest,
                dest_index
            );
        }
    }
}

// return a new copy of arr, with its stride coerced to strides
ndarray * ndarray_coerce_stride(const ndarray * arr, index_t * strides) {
    ndarray * output = malloc(sizeof(ndarray));
    output->ndims = arr->ndims;
    output->shape = array_index_t_copy(arr->shape, arr->ndims);
    output->strides = array_index_t_copy(strides, arr->ndims);
  
    if (arr->ndims == 0) {
        output->data = malloc(ndarray_datasize(arr));
        *output->data = *(arr->data);
    } else {
        output->data = malloc(ndarray_datasize(arr));
        coerce_stride_recur(
            0, arr, 0, output, 0
        );
    }

    return output;
}

// returns a new copy of arr which is broadcasted to the supplied dimension
// assumes that ndims is <= arr->ndims and shape is compatible
// compatible means the smallest arr_x->ndims dimensions in shape
// must be equaivalent to arr_x->shape
// i.e:
//    forall i in 0..(arr_x->ndims -1): arr_x->shape[i] == shape[ndims - arr_x->ndims + i]
ndarray * ndarray_broadcast(const ndarray * arr, index_t ndims, index_t * shape) {
    // determine how many folds of copy we need to do
    ndarray * output;
    unsigned folds = 1;
    const unsigned dims_difference = ndims - arr->ndims;
    const index_t original_elements_count = ndarray_elements_count(arr);
    for (index_t i = 0 ; i < ndims - arr->ndims ; i++) {
        folds *= shape[i]; 
    }
    // folds immutable from this point on!
    
    // create new object, copy from old to new
    output = malloc(sizeof(ndarray));
    output->ndims = ndims;
    output->shape = array_index_t_copy(shape, ndims);
    output->strides = malloc(ndims * sizeof(index_t));
    output->data = malloc(folds * original_elements_count * sizeof(double));

    // populate data
    for (index_t f = 0 ; f < folds ; f++)  {
        for (index_t i = 0 ; i < original_elements_count ; i++) {
            output->data[f * original_elements_count + i] = arr->data[i];
        }
    }

    // populate strides
    for (index_t i = dims_difference ; i < ndims ; i++) {
        output->strides[i] = arr->strides[i-dims_difference];
    }
    
    if (dims_difference == 0) {
        return output;
    } else {
        output->strides[dims_difference - 1] = original_elements_count;
        // i may be negative in this loop, so use long long 
        for (long long i = ((long long) dims_difference) - 2 ; i >= 0 ; i--) {
            output->strides[i] = output->strides[i+1] * output->shape[i+1]; 
        }

        return output;
    }
}

// at this point, arr_x and arr_y are _assumed_ to be in compatible dimensions
// i.e: the openCL code knows how to handle them. It doesn't mean that arr_x and arr_y are of equal dimensions
void map_run_kernel(
        cl_command_queue cmd_queue,
        cl_kernel kernel,
        const ndarray * arr_x,
        const ndarray * arr_y,
        double * ret
    ) {

    cl_int status;
    const unsigned datasize = ndarray_datasize(arr_x);
    cl_mem buffer_output, buffer_x, buffer_y, buffer_shape_x, buffer_strides_x, buffer_shape_y, buffer_strides_y;
    index_t number_of_elements = ndarray_elements_count(arr_x);
    size_t global_work_size_dims;
    size_t * global_work_size = get_global_work_size(
        arr_x,
        arr_y,
        &global_work_size_dims
    );
    const index_t max_dims = MAX(arr_x->ndims, arr_y->ndims);

    number_of_elements = ndarray_elements_count(arr_x);
    TRY_AND_CATCH_ERROR(
        buffer_x = buffers_create(CL_MEM_READ_ONLY, datasize, NULL, &status),
        status
    );
    TRY_AND_CATCH_ERROR(
        buffer_y = buffers_create(CL_MEM_READ_ONLY, datasize, NULL, &status),
        status
    );
    TRY_AND_CATCH_ERROR(
        buffer_shape_x = buffers_create(
            CL_MEM_READ_ONLY,
            sizeof(index_t) * arr_x->ndims,
            NULL,
            &status
        );,
        status
    );
    TRY_AND_CATCH_ERROR(
        buffer_strides_x = buffers_create(
            CL_MEM_READ_ONLY,
            sizeof(index_t) * arr_x->ndims,
            NULL,
            &status
        );,
        status
    );
    TRY_AND_CATCH_ERROR(
        buffer_shape_y = buffers_create(
            CL_MEM_READ_ONLY,
            sizeof(index_t) * arr_y->ndims,
            NULL,
            &status
        );,
        status
    );
    TRY_AND_CATCH_ERROR(
        buffer_strides_y = buffers_create(
            CL_MEM_READ_ONLY,
            sizeof(index_t) * arr_y->ndims,
            NULL,
            &status
        );,
        status
    );
    TRY_AND_CATCH_ERROR(
        buffer_output = buffers_create(
            CL_MEM_WRITE_ONLY,
            datasize,
            NULL,
            &status
        );,
        status
    );
    TRY_AND_CATCH_ERROR(
        status = clEnqueueWriteBuffer(
            cmd_queue, buffer_x,
            CL_FALSE, 0, datasize,
            arr_x->data, 0, NULL, NULL
        );,
        status
    );
    TRY_AND_CATCH_ERROR(
        status = clEnqueueWriteBuffer(
            cmd_queue, buffer_shape_x,
            CL_FALSE, 0,
            sizeof(index_t) * arr_x->ndims,
            arr_x->shape, 0, NULL, NULL
        );,
        status
    );
    TRY_AND_CATCH_ERROR(
        status = clEnqueueWriteBuffer(
            cmd_queue, buffer_strides_x,
            CL_FALSE, 0,
            sizeof(index_t) * arr_x->ndims,
            arr_x->strides,
            0, NULL, NULL
        );,
        status
    );
    TRY_AND_CATCH_ERROR(
        status = clEnqueueWriteBuffer(
            cmd_queue, buffer_y,
            CL_FALSE, 0, datasize,
            arr_y->data, 0, NULL, NULL
        );,
        status
    );
    TRY_AND_CATCH_ERROR(
        status = clEnqueueWriteBuffer(
            cmd_queue, buffer_shape_y,
            CL_FALSE, 0,
            sizeof(index_t) * arr_y->ndims,
            arr_y->shape, 0, NULL, NULL
        );,
        status
    );
    TRY_AND_CATCH_ERROR(
        status = clEnqueueWriteBuffer(
            cmd_queue, buffer_strides_y,
            CL_FALSE, 0,
            sizeof(index_t) * arr_y->ndims,
            arr_y->strides, 0, NULL, NULL
        );,
        status
    );
    TRY_AND_CATCH_ERROR(
        status = clSetKernelArg(
            kernel, 0, sizeof(cl_mem),
            &buffer_x
        );,
        status
    );
    TRY_AND_CATCH_ERROR(
        status = clSetKernelArg(
            kernel, 1, sizeof(cl_mem),
            &buffer_shape_x
        );,
        status
    );
    TRY_AND_CATCH_ERROR(
        status = clSetKernelArg(
            kernel, 2, sizeof(cl_mem),
            &buffer_strides_x
        );,
        status
    );
    TRY_AND_CATCH_ERROR(
        status = clSetKernelArg(kernel, 3,
            sizeof(index_t),
            &arr_x->ndims
        );,
        status
    );
    TRY_AND_CATCH_ERROR(
        status = clSetKernelArg(
            kernel, 4, sizeof(cl_mem),
            &buffer_y
        );,
        status
    );
    TRY_AND_CATCH_ERROR(
        status = clSetKernelArg(
            kernel, 5, sizeof(cl_mem),
            &buffer_shape_y
        );,
        status
    );
    TRY_AND_CATCH_ERROR(
        status = clSetKernelArg(
            kernel, 6, sizeof(cl_mem),
            &buffer_strides_y
        );,
        status
    );
    TRY_AND_CATCH_ERROR(
        status = clSetKernelArg(
            kernel, 7, sizeof(index_t),
            &arr_y->ndims
        );,
        status
    );
    TRY_AND_CATCH_ERROR(
        status = clSetKernelArg(
            kernel, 8, sizeof(cl_mem),
            &buffer_output
        );,
        status
    );
    TRY_AND_CATCH_ERROR(
        status = clEnqueueNDRangeKernel(
            cmd_queue, kernel, global_work_size_dims,
            NULL, global_work_size, NULL, 0, NULL, NULL
        );,
        status
    );

    clEnqueueReadBuffer(
        cmd_queue,
        buffer_output,
        CL_TRUE,
        0,
        number_of_elements * sizeof(double),
        ret,
        0,
        NULL,
        NULL
    );

    // free memory
    free(global_work_size);
    clReleaseMemObject(buffer_x);
    clReleaseMemObject(buffer_shape_x);
    clReleaseMemObject(buffer_strides_x);
    clReleaseMemObject(buffer_y);
    clReleaseMemObject(buffer_shape_y);
    clReleaseMemObject(buffer_strides_y);
    clReleaseMemObject(buffer_output);
}

ndarray * map_helper(
        cl_command_queue cmd_queue,
        cl_kernel kernel,
        const ndarray * arr_x,
        const ndarray * arr_y
    ) {
    
    if (arr_x->ndims == arr_y->ndims) {
        double * data = malloc(sizeof(double) * ndarray_elements_count(arr_x));
        ndarray * output;

        if (arr_x->ndims <= 3) {
             map_run_kernel(cmd_queue, kernel, arr_x, arr_y, data);
        } else {
            // coerce the strides!
            ndarray * coerced = ndarray_coerce_stride(arr_y, arr_x->strides);
            map_run_kernel(cmd_queue, kernel, arr_x, coerced, data);
            ndarray_release(coerced);
        }

        output = malloc(sizeof(ndarray));
        output->data = data;
        output->ndims = arr_x->ndims;
        output->strides = array_index_t_copy(arr_x->strides, arr_x->ndims);
        output->shape = array_index_t_copy(arr_x->shape, arr_x->ndims);
        return output;

    } else if (arr_x->ndims > arr_y->ndims) {
        // broadcasting
        ndarray * broadcasted = ndarray_broadcast(arr_y, arr_x->ndims, arr_x->shape);
        ndarray * ret = map_helper(cmd_queue, kernel, arr_x, broadcasted);
        ndarray_release(broadcasted);
        return ret;
    } else { 
        // arr_x->ndims < arr-y->ndims
        return map_helper(cmd_queue, kernel, arr_y, arr_x);
    }
}

void map_scalar_run_kernel(
        cl_command_queue cmd_queue,
        cl_kernel kernel,
        const ndarray * arr_x,
        const double y,
        double * ret
    ) {

    cl_int status;
    cl_mem buffer_output, buffer_x;
    size_t datasize = ndarray_datasize(arr_x);
    unsigned number_of_elements = ndarray_elements_count(arr_x);
    size_t global_work_size[1] = { number_of_elements };

    buffer_x = buffers_create(CL_MEM_READ_ONLY,
            datasize, NULL, &status);
    buffer_output = buffers_create(CL_MEM_WRITE_ONLY,       
            datasize, NULL, &status);
    status = clEnqueueWriteBuffer(
        cmd_queue,
        buffer_x,
        CL_TRUE,
        0,
        datasize,
        (void*) arr_x->data,
        0,
        NULL,
        NULL
    );

    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_x);
    status |= clSetKernelArg(kernel, 1, sizeof(double), &y);
    status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &buffer_output);                  status = clEnqueueNDRangeKernel(cmd_queue, kernel, 1,   
            NULL, global_work_size, NULL, 0, NULL, NULL);
    clEnqueueReadBuffer(
        cmd_queue,
        buffer_output,
        CL_TRUE,
        0,
        number_of_elements * sizeof(double),
        ret,
        0,
        NULL,
        NULL
    );

    clReleaseMemObject(buffer_x);
    clReleaseMemObject(buffer_output);
}

ndarray * map_factory(const ndarray * arr_x, const ndarray * arr_y, kernel_type_t kernel_id) {
    cl_int status;
    cl_command_queue cmd_queue = clCreateCommandQueue(context_get(), device_get(), 0, &status);
    cl_kernel kernel = kernels_get(context_get(), device_get(), kernel_id);
    return map_helper(
        cmd_queue,
        kernel,
        arr_x, arr_y
    );
}

ndarray * map_scalar_factory(const ndarray * arr_x, double y, kernel_type_t kernel_id) {
    cl_int status;
    cl_command_queue cmd_queue = clCreateCommandQueue(context_get(), device_get(), 0, &status);
    ndarray * output = ndarray_clone_structure(arr_x);
    map_scalar_run_kernel(
        cmd_queue,
        kernels_get(context_get(), device_get(), kernel_id),
        arr_x,
        y,
        output->data
    );

    // free unused memory
    clReleaseCommandQueue(cmd_queue);

    return output;
}

void map_bang_helper(
        cl_command_queue cmd_queue,
        cl_kernel kernel,
        ndarray * arr_x,
        const ndarray * arr_y
    ) {

    if (arr_x->ndims == arr_y->ndims) {

        if (arr_x->ndims <= 3) {
            map_run_kernel(cmd_queue, kernel, arr_x, arr_y, arr_x->data);
        } else {
            // coerce the strides!
            ndarray * coerced = ndarray_coerce_stride(arr_y, arr_x->strides);
            map_run_kernel(cmd_queue, kernel, arr_x, coerced, arr_x->data);
            ndarray_release(coerced);
        }

    } else if (arr_x->ndims > arr_y->ndims) {
        // broadcasting
        ndarray * broadcasted = ndarray_broadcast(arr_y, arr_x->ndims, arr_x->shape);
        map_bang_helper(cmd_queue, kernel, arr_x, broadcasted);
        ndarray_release(broadcasted);
    } else { 
        // arr_x->ndims < arr-y->ndims
        // should not happen in map_bang_helper
        fprintf(
            stderr,
            "An error occured (arr_x->ndims < arr-y->ndims) in running gpu.matrix"
            "at line %u of %s\n" ,
            __LINE__, __FILE__
        );
        exit(1);
    }
}

// map_bang, we assume that arr_x is is compatible with arr_y in the sense
// that we do not need have to broadcast arr_x to suit arr_y
// hence, we will by pass map_helper, which handles coercing and broadcasting
void map_bang_factory(ndarray * arr_x, const ndarray * arr_y, kernel_type_t kernel_id) {
    cl_int status;
    cl_command_queue cmd_queue = clCreateCommandQueue(context_get(), device_get(), 0, &status);
    cl_kernel kernel = kernels_get(
        context_get(),
        device_get(),
        kernel_id
    );

    map_bang_helper(cmd_queue, kernel, arr_x, arr_y);

    // free memory
    clReleaseCommandQueue(cmd_queue);
}

void map_scalar_bang_factory(ndarray * arr_x, double y, kernel_type_t kernel_id) {
    cl_int status;
    cl_command_queue cmd_queue = clCreateCommandQueue(context_get(), device_get(), 0, &status);
    map_scalar_run_kernel(
        cmd_queue,
        kernels_get(context_get(), device_get(), kernel_id),
        arr_x,
        y,
        arr_x->data
    );
    
    clReleaseCommandQueue(cmd_queue);
}

// API to do NDArray ops

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
    output->strides  = array_index_t_copy(arr_x->strides, arr_x->ndims);
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

void ndarray_set_nd(ndarray * arr, const index_t * indexes, double v) {
    long idx = 0;
    for (long i = 0 ; i < arr->ndims ; i++) {
        idx += arr->strides[i] * indexes[i];
    }
    arr->data[idx] = v;
}

bool ndarray_equals(const ndarray * arr_x, const ndarray * arr_y) {
    if (arr_x->ndims != arr_y->ndims ||
            !array_index_t_is_equal(arr_x->shape, arr_y->shape, arr_x->ndims)) {
        return false;
    } else {
        if (array_index_t_is_equal(arr_x->strides, arr_y->strides, arr_x->ndims)) {
            return array_double_is_equal(
                arr_x->data,
                arr_y->data,
                ndarray_elements_count(arr_x)
            );
        } else {
            ndarray * coerced = ndarray_coerce_stride(arr_y, arr_x->strides);
            bool flag = array_double_is_equal(
                arr_x->data,
                arr_y->data,
                ndarray_elements_count(arr_x)
            );
            ndarray_release(coerced);
            return flag;
        }
    }

    return false;
}

bool ndarray_equals_scalar(const ndarray * arr, const double y) {
    if (arr->ndims != 0) {
        return false;
    } else {
        return (*arr->data) == y;
    }
    // to prevent clang error
    return false;
}

void ndarray_flatten_recur(
    const index_t dim,              // the current dimension
    const ndarray * src,
    const index_t src_index_accum,
    double * dest,
    index_t * index) {

    for (index_t i = 0 ; i < src->shape[dim] ; i++) {
        const index_t src_index = src_index_accum + i * src->strides[dim];

        if (dim == src->ndims - 1) {
            dest[*index] = src->data[src_index];
            (*index)++;
        } else {
            ndarray_flatten_recur(
                dim + 1,
                src,
                src_index,
                dest,
                index
            );
        }
    }
}

double * ndarray_flatten(const ndarray * arr) {
    double * data = malloc(ndarray_elements_count(arr) * sizeof(double));
    index_t idx = 0;
    ndarray_flatten_recur(0, arr, 0, data, &idx);
    return data;
}

// Arimethic ops

ndarray * ndarray_add(const ndarray * arr_x, const ndarray * arr_y) {
    return map_factory(arr_x, arr_y, KERNEL_ADD);
}

ndarray * ndarray_add_scalar(const ndarray * arr_x, const double y) {
    return map_scalar_factory(arr_x, y, KERNEL_ADD_SCALAR);
}

void ndarray_add_bang(ndarray * arr_x, const ndarray * arr_y) {
    map_bang_factory(arr_x, arr_y, KERNEL_ADD);
}

void ndarray_add_scalar_bang(ndarray * arr_x, const double y) {
    map_scalar_bang_factory(arr_x, y, KERNEL_ADD_SCALAR);
}
 
ndarray * ndarray_sub(const ndarray * arr_x, const ndarray * arr_y) {
    return map_factory(arr_x, arr_y, KERNEL_SUB);
}

ndarray * ndarray_sub_scalar(const ndarray * arr_x, const double y) {
    return map_scalar_factory(arr_x, y, KERNEL_SUB_SCALAR);
}

void ndarray_sub_bang(ndarray * arr_x, const ndarray * arr_y) {
    map_bang_factory(arr_x, arr_y, KERNEL_SUB);
}

void ndarray_sub_scalar_bang(ndarray * arr_x, const double y) {
    map_scalar_bang_factory(arr_x, y, KERNEL_SUB_SCALAR);
}
 
ndarray * ndarray_mul(const ndarray * arr_x, const ndarray * arr_y) {
    return map_factory(arr_x, arr_y, KERNEL_MUL);
}

ndarray * ndarray_mul_scalar(const ndarray * arr_x, const double y) {
    return map_scalar_factory(arr_x, y, KERNEL_MUL_SCALAR);
}

void ndarray_mul_bang(ndarray * arr_x, const ndarray * arr_y) {
    map_bang_factory(arr_x, arr_y, KERNEL_MUL);
}

void ndarray_mul_scalar_bang(ndarray * arr_x, const double y) {
    map_scalar_bang_factory(arr_x, y, KERNEL_MUL_SCALAR);
}
 
ndarray * ndarray_div(const ndarray * arr_x, const ndarray * arr_y) {
    return map_factory(arr_x, arr_y, KERNEL_DIV);
}

ndarray * ndarray_div_scalar(const ndarray * arr_x, const double y) {
    return map_scalar_factory(arr_x, y, KERNEL_DIV_SCALAR);
}

void ndarray_div_bang(ndarray * arr_x, const ndarray * arr_y) {
    map_bang_factory(arr_x, arr_y, KERNEL_DIV);
}

void ndarray_div_scalar_bang(ndarray * arr_x, const double y) {
    map_scalar_bang_factory(arr_x, y, KERNEL_DIV_SCALAR);
}
