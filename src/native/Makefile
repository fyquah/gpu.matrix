TARGET_FILE=out/main
FLAGS=-O2 -fPIC -std=c99 -lOpenCL -I/usr/lib/jvm/java-1.7.0-openjdk-amd64/include/
OBJECT_FILES=ndarray.o utils/array.o utils/cl_helper.o utils/files.o \
	     utils/kernels.o ndarray_jni.o

all: ${TARGET_FILE} 

libs: ${OBJECT_FILES}
	gcc $^ -o libgpu-matrix-ndarray.so \
	    ${FLAGS} \
	    -shared

clean:
	rm -rf ${OBJECT_FILES}

%.o: %.c
	gcc -c -o $@ $< $(FLAGS)

${TARGET_FILE}: main.c ${OBJECT_FILES}
	gcc -o $@ $^ ${FLAGS}

