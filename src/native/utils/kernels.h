#ifndef KERNELS_H
#define KERNELS_H

#include <string.h>
#include <stdio.h>

#include "CL/cl.h"
#include "files.h"

typedef const unsigned kernel_type_t;

#define KERNEL_ADD 0
#define KERNEL_ADD_SCALAR 1
#define KERNEL_ADD_BANG 2
#define KERNEL_ADD_SCALAR_BANG 3
#define KERNEL_SUB 4
#define KERNEL_SUB_SCALAR 5
#define KERNEL_SUB_BANG 6
#define KERNEL_SUB_SCALAR_BANG 7
#define KERNEL_MUL 8
#define KERNEL_MUL_SCALAR 9
#define KERNEL_MUL_BANG 10
#define KERNEL_MUL_SCALAR_BANG 11
#define KERNEL_DIV 12
#define KERNEL_DIV_SCALAR 13
#define KERNEL_DIV_BANG 14
#define KERNEL_DIV_SCALAR_BANG 15

const char * get_program_file_name(kernel_type_t);
const char * get_cl_function_name(kernel_type_t);
cl_kernel kernels_get(cl_context, cl_device_id, kernel_type_t);

#endif
