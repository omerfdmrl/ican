#ifndef ICAN_ARRAY_H

#define ICAN_ARRAY_H

#include "ican.h"

struct s_array1d {
    int64 rows;
    float *data;
};

struct s_array2d {
    int64 rows;
    int64 cols;
    float *data;
};

typedef struct s_array1d Array1D;
typedef struct s_array2d Array2D;

#ifndef ICAN_ARRAY_SIMD
#define ICAN_ARRAY_SIMD 8
#endif // !ICAN_ARRAY_SIMD

#if ICAN_ARRAY_SIMD == 64
    #define ICAN_ARRAY_SIMD_align 64
    #define ICAN_ARRAY_SIMD_load_ps _mm512_load_ps
    #define ICAN_ARRAY_SIMD_store_ps _mm512_store_ps
    #define ICAN_ARRAY_SIMD_add_ps _mm512_add_ps
    #define ICAN_ARRAY_SIMD_sub_ps _mm512_sub_ps
    #define ICAN_ARRAY_SIMD_mul_ps _mm512_mul_ps
    #define ICAN_ARRAY_SIMD_hadd_ps _mm512_hadd_ps
    #define ICAN_ARRAY_SIMD_set1_ps _mm512_set1_ps
    #define ICAN_ARRAY_SIMD_setzero_ps _mm512_setzero_ps
#elif ICAN_ARRAY_SIMD == 32
    #define ICAN_ARRAY_SIMD_align 32
    #define ICAN_ARRAY_SIMD_load_ps _mm256_load_ps
    #define ICAN_ARRAY_SIMD_store_ps _mm256_store_ps
    #define ICAN_ARRAY_SIMD_add_ps _mm256_add_ps
    #define ICAN_ARRAY_SIMD_sub_ps _mm256_sub_ps
    #define ICAN_ARRAY_SIMD_mul_ps _mm256_mul_ps
    #define ICAN_ARRAY_SIMD_hadd_ps _mm256_hadd_ps
    #define ICAN_ARRAY_SIMD_set1_ps _mm256_set1_ps
    #define ICAN_ARRAY_SIMD_setzero_ps _mm256_setzero_ps
#elif ICAN_ARRAY_SIMD == 16
    #define ICAN_ARRAY_SIMD_align 64
    #define ICAN_ARRAY_SIMD_load_ps _mm512_load_ps
    #define ICAN_ARRAY_SIMD_store_ps _mm512_store_ps
    #define ICAN_ARRAY_SIMD_add_ps _mm512_add_ps
    #define ICAN_ARRAY_SIMD_sub_ps _mm512_sub_ps
    #define ICAN_ARRAY_SIMD_mul_ps _mm512_mul_ps
    #define ICAN_ARRAY_SIMD_hadd_ps _mm512_hadd_ps
    #define ICAN_ARRAY_SIMD_set1_ps _mm512_set1_ps
    #define ICAN_ARRAY_SIMD_setzero_ps _mm512_setzero_ps
#elif ICAN_ARRAY_SIMD == 8
    #define ICAN_ARRAY_SIMD_align 32
    #define ICAN_ARRAY_SIMD_load_ps _mm256_load_ps
    #define ICAN_ARRAY_SIMD_store_ps _mm256_store_ps
    #define ICAN_ARRAY_SIMD_add_ps _mm256_add_ps
    #define ICAN_ARRAY_SIMD_sub_ps _mm256_sub_ps
    #define ICAN_ARRAY_SIMD_mul_ps _mm256_mul_ps
    #define ICAN_ARRAY_SIMD_hadd_ps _mm256_hadd_ps
    #define ICAN_ARRAY_SIMD_set1_ps _mm256_set1_ps
    #define ICAN_ARRAY_SIMD_setzero_ps _mm256_setzero_ps
#elif ICAN_ARRAY_SIMD == 4
    #define ICAN_ARRAY_SIMD_align 16
    #define ICAN_ARRAY_SIMD_load_ps _mm128_load_ps
    #define ICAN_ARRAY_SIMD_store_ps _mm128_store_ps
    #define ICAN_ARRAY_SIMD_add_ps _mm128_add_ps
    #define ICAN_ARRAY_SIMD_sub_ps _mm128_sub_ps
    #define ICAN_ARRAY_SIMD_mul_ps _mm128_mul_ps
    #define ICAN_ARRAY_SIMD_hadd_ps _mm128_hadd_ps
    #define ICAN_ARRAY_SIMD_set1_ps _mm128_set1_ps
    #define ICAN_ARRAY_SIMD_setzero_ps _mm128_setzero_ps
#endif

Array2D *array2d_alloc(int64 rows, int64 cols);
void array2d_free(Array2D *array);
Array2D *array2d_add(Array2D *A, Array2D *B);
Array2D *array2d_sub(Array2D *A, Array2D *B);
Array2D *array2d_dot(Array2D *A, Array2D *B);
Array2D *array2d_apply(Array2D *A, float (*func)(float));
Array2D *array2d_sort(Array2D *A, int (*comp)(float*, float*, int64));
Array2D *array2d_scale(Array2D *A, float scalar);
void array2d_fill(Array2D *A, float data);
float array2d_max(Array2D *A);
float array2d_min(Array2D *A);
Array2D *array2d_slice(Array2D *A, int64 row_start, int64 row_end, int64 col_start, int64 col_end);
Array2D *array2d_clone(Array2D *A);
Array2D *array2d_transpose(Array2D *A);
Array2D *array2d_from(float *data, int64 rows, int64 cols);
Array1D *array2d_to_array1d(Array2D *A);
void array2d_print(Array2D *array);

Array1D *array1d_alloc(int64 rows);
void array1d_free(Array1D *array);
Array1D *array1d_add(Array1D *A, Array1D *B);
Array1D *array1d_sub(Array1D *A, Array1D *B);
float array1d_sum(Array1D *A);
Array1D *array1d_dot(Array1D *A, Array1D *B);
float array1d_dot_scalar(Array1D *A, Array1D *B);
void array1d_fill(Array1D *A, float data);
Array1D *array1d_apply(Array1D *A, float (*func)(float));
Array1D *array1d_sort(Array1D *A, int (*comp)(float, float));
Array1D *array1d_scale(Array1D *A, float scalar);
float array1d_max(Array1D *A);
float array1d_min(Array1D *A);
Array1D *array1d_slice(Array1D *A, int64 start, int64 end);
Array1D *array1d_clone(Array1D *A);
Array1D *array1d_from(float *data, int64 rows);
Array2D *array1d_to_array2d(Array1D *A, int64 rows, int64 cols);
void array1d_print(Array1D *array);

#endif // !ICAN_ARRAY_H