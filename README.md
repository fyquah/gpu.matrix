# gpu.matrix

A GPU implementation of core.matrix. Very strong emphasis on asynchrnous data processing (aka, process the data on the background
when the user is doing something else, blocks only when the user requests for the data).

# Code Structure

Should be pretty self-explanatory:

~~~bash

native/
# The C CODE
java/
# The java class declarations. Source code here is almost empty 
# most declaration are simply calls to native methods.
# Matrices' type-checking (i.e dims checking) are done 
# in clojure instead
clojure/

~~~

# Classes

Right now, there are 3 main classes:

* **NDArray** - NDimensional arrays
* **Vector**
* **Matrix**

All numbers are stored internally as `double`, overflows will not be checked. This should suffice most general matrix computations usages.

# Loading Native Libraries

During development, `libgpu-matrix.(so|dll|dylib)` will be loaded from the `src/native/` directory.
In production, the shared library will be loaded dynamically from the shared resource using the
`gpu.matrix.LoaderUtils.loadLibrary ` static method. This methods resolves the relevant library from
`java.library.path` and `resources/native/<OS_NAME>/<ARCH_NAME>/` in the mentioned order. Throws an 
`UnsatisfiedLinkError` if the library cannot be found.

# Conventions

Conventions for native and clojure code are in the README in the relevant source code directories.
