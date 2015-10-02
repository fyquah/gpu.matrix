UNAME_S := $(shell uname -s)
FLAGS=-O2 -fPIC -std=c99 -lm
.DEFAULT_GOAL := all

ifeq ($(UNAME_S),Linux)
    FLAGS += -D LINUX -lOpenCL -I/usr/lib/jvm/java-1.7.0-openjdk-amd64/include/
    TARGET_LIB=libgpu-matrix.so
    CC=gcc
endif
ifeq ($(UNAME_S),Darwin)
    FLAGS += -D DARWIN -framework opencl -I/System/Library/Frameworks/JavaVM.framework/Headers
    TARGET_LIB=libgpu-matrix.dylib
    CC=clang
endif

OUT_DIR=out
TARGET_FILE=${OUT_DIR}/main

OBJECT_FILES=ndarray.o utils/array.o utils/cl_helper.o utils/files.o \
	     utils/kernels.o
JNI_OBJECT_FILES=ndarray_jni.o initializer_jni.o

# Rules:

all: ${TARGET_FILE} 

libs: ${TARGET_LIB}

${TARGET_LIB}: ${OBJECT_FILES} ${JNI_OBJECT_FILES}
	${CC} $^ -o $@ ${FLAGS} -shared

clean:
	rm -rf ${OBJECT_FILES} ${JNI_OBJECT_FILES} \
	    *.so out/*

${OUT_DIR}:
	mkdir -p ${OUT_DIR}

%.o: %.c
	${CC} -c -o $@ $< $(FLAGS)

${TARGET_FILE}: main.c ${OBJECT_FILES}
	${CC} -o $@ $^ ${FLAGS}

