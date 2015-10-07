This directory contains all the native code for `gpu.matrix`.

You might notice that no shape / pointer checking is done. This is intentional - checking should be done by clojure. This is to prevent unnecessary checking where data is guranteed to be okay.

# Compile

~~~bash
make
~~~

# Install to resources/

~~~bash
# The Makefile will recognize your machine's OS and ARCH
make install
~~~

# Code Conventions

This is not an exhaustive (nor final) list of conventions. I am also open to improvements where possible.

## DRY-ing up

I realize that a lot of code is repetitive. I can think of a few ways around this:

* Use constant arguments. This will impose some (negligible?) run time overhead
* Define pretty messy macros
* Generate code with perl scripts

## Function Names

"Object" method calls are usually in the form:

~~~C
gpu_matrix_<CLASS_NAME>_<method_name>(this_pointer, args ...);

# eg:

vector * v_x = malloc(sizeof(vector));
// some initialization
vector * v_x = gpu_matrix_vector_axpy(v_x, 1.0, v_x);

~~~

## Function return signature

If the function modifies the "this" argument, the name should end with a BANG. eg: `add_scalar_BANG`.

If it has to dynamically allocate memory to a new object, return that object rather than set its value using a `void` function.

~~~C

vector * gpu_matrix_vector_add(vector*, vector*);
vector * v_x = gpu_matrix_vector_add(v_y, v_y);

// instead of

void gpu_matrix_vector_add(vector*, vector*, vector*);
gpu_matrix_vector_add(v_y, v_y, v_x);

~~~

## Unit Tests

I can considering several test frameworks. If I can't find anything I like, I will just write one.
