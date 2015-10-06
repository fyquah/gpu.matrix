#include "profiling.h"

cl_ulong get_event_time(cl_event event, char * description) {
    cl_ulong end, start;

    clGetEventProfilingInfo(
        event,
        CL_PROFILING_COMMAND_START,
        sizeof(cl_ulong),
        &start,
        NULL
    );
    clGetEventProfilingInfo(
        event,
        CL_PROFILING_COMMAND_END,
        sizeof(cl_ulong),
        &end,
        NULL
    );

#ifdef ENABLE_PROFILING
    printf("%s: %.3f milliseconds\n", description, ((double) (end-start) * 0.000001));
#endif

    return end-start;
}
