#include "CL/cl.h"

typedef struct {
   cl_context context;
   cl_platform_id platform;
   cl_device_id device;
   cl_command_queue cmd_queue;
} opencl_config_t;

opencl_config_t get_common_config();

cl_context context_get();
cl_device_id device_get();
void init();
void destroy();
cl_mem buffers_create(cl_mem_flags, size_t, void*, cl_int*);
cl_command_queue command_queue_create(cl_command_queue_properties properties, cl_int * errcode_res);
