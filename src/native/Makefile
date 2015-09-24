TARGET_FILE=out/main
FLAGS=-O2 -fPIC -std=c99 -lOpenCL -I/usr/lib/jvm/java-1.7.0-openjdk-amd64/include/
OBJECT_FILES=ndarray.o utils/array.o utils/cl_helper.o utils/files.o \
	     utils/kernels.o
JNI_OBJECT_FILES=ndarray_jni.o jvmloader_jni.o
CC=gcc

all: ${TARGET_FILE} 

libs: libgpu-matrix.so

libgpu-matrix.so: ${OBJECT_FILES} ${JNI_OBJECT_FILES}
	${CC} $^ -o $@ ${FLAGS} -shared

clean:
	rm -rf ${OBJECT_FILES} ${JNI_OBJECT_FILES}
	rm -rf *.so

%.o: %.c
	${CC} -c -o $@ $< $(FLAGS)

${TARGET_FILE}: main.c ${OBJECT_FILES}
	${CC} -o $@ $^ ${FLAGS}

