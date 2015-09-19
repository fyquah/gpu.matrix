UNAME_S := $(shell uname -s)
FLAGS=-O2 -fPIC -std=c99 
CC=gcc
.DEFAULT_GOAL := all

ifeq ($(UNAME_S),Linux)
    FLAGS += -D LINUX -lOpenCL -I/usr/lib/jvm/java-1.7.0-openjdk-amd64/include/
endif
ifeq ($(UNAME_S),Darwin)
    FLAGS += -D DARWIN -framework opencl -I/System/Library/Frameworks/JavaVM.framework/Headers
endif

OUT_DIR=out
TARGET_FILE=${OUT_DIR}/main

OBJECT_FILES=ndarray.o utils/array.o utils/cl_helper.o utils/files.o \
	     utils/kernels.o ndarray_jni.o

# Rules:

all: ${TARGET_FILE} 

libs: ${OBJECT_FILES}
	${CC} $^ -o libgpu-matrix-ndarray.so \
	    ${FLAGS} \
	    -shared

clean:
	rm -rf ${OBJECT_FILES}
	rm -rf out/*

${OUT_DIR}:
	mkdir -p ${OUT_DIR}

%.o: %.c
	${CC} -c -o $@ $< $(FLAGS)

${TARGET_FILE}: main.c ${OBJECT_FILES}
	${CC} -o $@ $^ ${FLAGS}

