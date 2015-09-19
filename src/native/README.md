# Structure

**ndarray.c**
**ndarray.h**
**ndarray.java**
**ndarray.clj**

Implementation, header files, JNI interface and clojure protocols interfaces for ndarray

**utils/**
- *array.c*
- *array.h*

General array helpers (copy, equals etc.)

- *cl\_helper.h*
- *cl\_helper.c*

OpenCL Helpers, create program, device, context etc.

- *files.h*
- *files.c*

File helpers, things like slurp or spit

- *kernels.c*
- *kernels.h*

Kernel helpers, things to return kernels, prevent them from being compiled multiple times. A number of kernels are store in
memory, and there is a limited number of kernel page. They will be managed using LRU, similiar to those for page replacement.

**scripts/**

Perl / shell scripts to generate source files

**opencl/**

OpenCL Modules go here
