#include "CL/cl.h"

cl_kernel kernels_get(const char *);
const char * get_program_file_name(const char *);
void init();
void destroy();
cl_mem buffers_create(cl_mem_flags, size_t, void*, cl_int*);
cl_command_queue command_queue_create(cl_command_queue_properties properties, cl_int * errcode_res);
