#include <sys/resource.h>
#include <time.h>
#define ICAN_USE_ARRAY
#include "ican/ican.h"

void print_memory_usage() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    printf("Memory usage: %ld KB\n", usage.ru_maxrss);
}

#define START_TIMER clock_t start = clock()
#define END_TIMER printf("Time: %f seconds\n", (double)(clock() - start) / CLOCKS_PER_SEC)

void benchmark_add() {
    printf("\n--- Benchmark: Add ---\n");
    Array2D *A = array2d_alloc(1000, 1000);
    Array2D *B = array2d_alloc(1000, 1000);
    array2d_fill(A, 1.0);
    array2d_fill(B, 2.0);

    print_memory_usage();
    START_TIMER;
    Array2D *C = array2d_add(A, B);
    END_TIMER;
    print_memory_usage();

    array2d_free(A);
    array2d_free(B);
    array2d_free(C);
}

void benchmark_dot() {
    printf("\n--- Benchmark: Dot ---\n");
    Array2D *A = array2d_alloc(1000, 1000);
    Array2D *B = array2d_alloc(1000, 1000);
    array2d_fill(A, 1.0);
    array2d_fill(B, 2.0);

    print_memory_usage();
    START_TIMER;
    Array2D *C = array2d_dot(A, B);
    END_TIMER;
    print_memory_usage();

    array2d_free(A);
    array2d_free(B);
    array2d_free(C);
}

float apply(float x) {
    return x * x;
}

void benchmark_apply() {
    printf("\n--- Benchmark: Apply ---\n");
    Array2D *A = array2d_alloc(1000, 1000);
    array2d_fill(A, 1.0);

    print_memory_usage();
    START_TIMER;
    Array2D *B = array2d_apply(A, apply);
    END_TIMER;
    print_memory_usage();

    array2d_free(A);
    array2d_free(B);
}

void benchmark_scale() {
    printf("\n--- Benchmark: Scale ---\n");
    Array2D *A = array2d_alloc(1000, 1000);
    array2d_fill(A, 1.0);

    print_memory_usage();
    START_TIMER;
    Array2D *B = array2d_scale(A, 0.5f);
    END_TIMER;
    print_memory_usage();

    array2d_free(A);
    array2d_free(B);
}

void benchmark_max() {
    printf("\n--- Benchmark: Max ---\n");
    Array2D *A = array2d_alloc(1000, 1000);
    array2d_fill(A, 1.0);

    print_memory_usage();
    START_TIMER;
    float B = array2d_max(A);
    END_TIMER;
    print_memory_usage();

    array2d_free(A);
    (void) B;
}

void benchmark_min() {
    printf("\n--- Benchmark: Min ---\n");
    Array2D *A = array2d_alloc(1000, 1000);
    array2d_fill(A, 1.0);

    print_memory_usage();
    START_TIMER;
    float B = array2d_min(A);
    END_TIMER;
    print_memory_usage();

    array2d_free(A);
    (void) B;
}

void benchmark_transpose() {
    printf("\n--- Benchmark: Transpose ---\n");
    Array2D *A = array2d_alloc(1000, 1000);
    array2d_fill(A, 1.0);

    print_memory_usage();
    START_TIMER;
    Array2D *B = array2d_transpose(A);
    END_TIMER;
    print_memory_usage();

    array2d_free(A);
    array2d_free(B);
}

// Ana fonksiyon
int main() {
    benchmark_add();
    benchmark_dot();
    benchmark_apply();
    benchmark_scale();
    benchmark_max();
    benchmark_min();
    return 0;
}
