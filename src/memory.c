#ifndef MEMORY_H
#define MEMORY_H

#include "ican.h"

MemoryBlock *memory_list = NULL;

void *memory_malloc(size_t size, const char *file, int line) {
    void *ptr = malloc(size);
    if (!ptr) return NULL;

    MemoryBlock *block = (MemoryBlock *)malloc(sizeof(MemoryBlock));
    block->file = file;
    block->is_freed = false;
    block->line = line;
    block->ptr = ptr;
    block->size = size;
    block->next = memory_list;
    memory_list = block;
    return ptr;
}

void memory_free(void *ptr, const char *file, int line) {
    MemoryBlock *current = memory_list;
    MemoryBlock *prev = NULL;

    while (current != NULL) {
        if (current->ptr == ptr) {
            if (current->is_freed) {
                printf("Double free detected in %s:%d\n", file, line);
                return;
            }
            current->is_freed = true;
            free(ptr);
            return;
        }
        prev = current;
        current = current->next;
    }
}

void *memory_calloc(size_t count, size_t size, const char *file, int line) {
    void *ptr = calloc(count, size);
    if (!ptr) return NULL;

    MemoryBlock *block = (MemoryBlock *)malloc(sizeof(MemoryBlock));
    block->file = file;
    block->is_freed = false;
    block->line = line;
    block->ptr = ptr;
    block->size = count * size;
    block->next = memory_list;
    memory_list = block;
    return ptr;
}

void memory_leaks() {
    MemoryBlock *current = memory_list;
    while (current != NULL) {
        if (!current->is_freed) {
            printf("Memory leak detected at %s:%d, size: %zu bytes\n", current->file, current->line, current->size);
        }
        current = current->next;
    }
}

#endif // !MEMORY_H