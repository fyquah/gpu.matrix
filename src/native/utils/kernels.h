#ifndef KERNELS_H
#define KERNELS_H

#include <assert.h>
#include <string.h>
#include <stdio.h>

#include <jni.h>
#include "../libs/cl.h" 
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
#define KERNEL_MMUL 8
#define KERNEL_VECTOR_AXPY 9
#define KERNEL_VECTOR_ASUM 10
#define KERNEL_VECTOR_MUL 11
#define KERNELS_COUNT 12

// Compiles the program and cahce it in a global buffer
static cl_program compile_program(cl_context, cl_device_id);

// Compiles a kernel and cache it in a global buffer, will run
// compile_prgram if a instance of a non-null program buffer 
// cannot be found
static cl_kernel compile_kernel(cl_context, cl_device_id, kernel_type_t);

// retrieves the program, runs compilation if a program cannot be found
cl_program program_get(cl_context, cl_device_id);

// retrieve a kernel, as defined by the kernel_id. If a kernel cannot
// be found, it will be compiled
cl_kernel kernels_get(cl_context, cl_device_id, kernel_type_t);

// set the global jvm buffer
void gpu_matrix_kernel_set_jvm(JNIEnv *);

// get file contents of the name specified by the directory
// given the arg <filename>:
// (if (defined? jvm-pointer)
//   (str jvm-resource-path "opencl/" filename)
//   (str SOURCE_PREFIX "opencl/" filename))
static char * get_file_contents(const char *);

// Get the relevant function / kernel name for a given openCL kernel_type_t
const char * get_cl_function_name(kernel_type_t);

// load compilation options
static char * get_compilation_options();

// TODO:
// Precompiles all the specified kernels
void gpu_matrix_kernel_precompile();

#endif
