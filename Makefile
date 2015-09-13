TARGET_FILE=out/main
FLAGS=-O2 -std=c99 -lOpenCL
SOURCE_FILES=main.c mul.c utils.c

all: ${SOURCE_FILES} 
	gcc $^ -o ${TARGET_FILE} ${FLAGS}
