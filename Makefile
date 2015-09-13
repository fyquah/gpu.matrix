TARGET_FILE=out/main
FLAGS=-O2 -std=c99 -lOpenCL
SOURCE_FILES=main.c ndarray.c utils/array.c utils/cl_helper.c utils/files.c \
	     utils/kernels.c

all: ${SOURCE_FILES} 
	gcc $^ -o ${TARGET_FILE} ${FLAGS}
