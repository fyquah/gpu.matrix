#include <stdlib.h>
#include <stdio.h>
#include "files.h"

char * slurp(const char * filename) {
    FILE * program_handle;
    unsigned long program_size;
    char * program_buffer;

    program_handle = fopen(filename, "r");
    fseek(program_handle, 0, SEEK_END);
    program_size = ftell(program_handle);
    rewind(program_handle);

    program_buffer = (char*) malloc(program_size + 1);
    program_buffer[program_size] = '\0';
    fread(program_buffer, sizeof(char), program_size, program_handle);
    fclose(program_handle);

    return program_buffer;
}
