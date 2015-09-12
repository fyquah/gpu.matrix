TARGET_FILE=out/main
FLAGS=-O2 -std=c99 -lOpenCL
all: main.c
	gcc $<  -o ${TARGET_FILE} ${FLAGS}
