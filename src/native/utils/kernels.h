#ifndef KERNELS_H
#define KERNELS_H

#include <string.h>
#include <stdio.h>

#include "CL/cl.h"
#include "files.h"

typedef const unsigned kernel_type_t;

#define KERNEL_ADD 0
#define KERNEL_ADD_SCALAR 1
#define KERNEL_SUB 2
#define KERNEL_SUB_SCALAR 3
#define KERNEL_MUL 4
#define KERNEL_MUL_SCALAR 5
#define KERNEL_DIV 6
#define KERNEL_DIV_SCALAR 7

const char * get_program_file_name(kernel_type_t);
const char * get_cl_function_name(kernel_type_t);
cl_kernel kernels_get(cl_context, cl_device_id, kernel_type_t);

#endif
