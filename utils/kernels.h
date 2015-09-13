#include <string.h>

#include "CL/cl.h"
#include "files.h"

cl_kernel kernels_get(cl_context, cl_device_id, const char *);
const char * get_program_file_name(const char *);
