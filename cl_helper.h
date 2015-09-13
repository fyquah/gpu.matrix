#include "CL/cl.h"

typedef struct {
   cl_context context;
   cl_platform_id platform;
   cl_device_id device;
   cl_command_queue cmd_queue;
} opencl_config_t;

opencl_config_t get_common_config();
