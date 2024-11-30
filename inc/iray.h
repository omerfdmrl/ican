#ifndef ICAN_IRAY_H

#define ICAN_IRAY_H

#include "ican.h"

struct s_iray1d {
    int64 rows;
    float *data;
};

struct s_iray2d {
    int64 rows;
    int64 cols;
    float *data;
};

typedef struct s_iray1d Iray1D;
typedef struct s_iray2d Iray2D;

#ifndef ICAN_IRAY_SIMD
#define ICAN_IRAY_SIMD 8
#endif // !ICAN_IRAY_SIMD

#if ICAN_IRAY_SIMD == 64
    #define ICAN_IRAY_SIMD_align 64
    #define ICAN_IRAY_SIMD_load_ps _mm512_load_ps
    #define ICAN_IRAY_SIMD_store_ps _mm512_store_ps
    #define ICAN_IRAY_SIMD_add_ps _mm512_add_ps
    #define ICAN_IRAY_SIMD_mul_ps _mm512_mul_ps
    #define ICAN_IRAY_SIMD_hadd_ps _mm512_hadd_ps
    #define ICAN_IRAY_SIMD_set1_ps _mm512_set1_ps
#elif ICAN_IRAY_SIMD == 32
    #define ICAN_IRAY_SIMD_align 32
    #define ICAN_IRAY_SIMD_load_ps _mm256_load_ps
    #define ICAN_IRAY_SIMD_store_ps _mm256_store_ps
    #define ICAN_IRAY_SIMD_add_ps _mm256_add_ps
    #define ICAN_IRAY_SIMD_mul_ps _mm256_mul_ps
    #define ICAN_IRAY_SIMD_hadd_ps _mm256_hadd_ps
    #define ICAN_IRAY_SIMD_set1_ps _mm256_set1_ps
#elif ICAN_IRAY_SIMD == 16
    #define ICAN_IRAY_SIMD_align 64
    #define ICAN_IRAY_SIMD_load_ps _mm512_load_ps
    #define ICAN_IRAY_SIMD_store_ps _mm512_store_ps
    #define ICAN_IRAY_SIMD_add_ps _mm512_add_ps
    #define ICAN_IRAY_SIMD_mul_ps _mm512_mul_ps
    #define ICAN_IRAY_SIMD_hadd_ps _mm512_hadd_ps
    #define ICAN_IRAY_SIMD_set1_ps _mm512_set1_ps
#elif ICAN_IRAY_SIMD == 8
    #define ICAN_IRAY_SIMD_align 32
    #define ICAN_IRAY_SIMD_load_ps _mm256_load_ps
    #define ICAN_IRAY_SIMD_store_ps _mm256_store_ps
    #define ICAN_IRAY_SIMD_add_ps _mm256_add_ps
    #define ICAN_IRAY_SIMD_mul_ps _mm256_mul_ps
    #define ICAN_IRAY_SIMD_hadd_ps _mm256_hadd_ps
    #define ICAN_IRAY_SIMD_set1_ps _mm256_set1_ps
#elif ICAN_IRAY_SIMD == 4
    #define ICAN_IRAY_SIMD_align 16
    #define ICAN_IRAY_SIMD_load_ps _mm128_load_ps
    #define ICAN_IRAY_SIMD_store_ps _mm128_store_ps
    #define ICAN_IRAY_SIMD_add_ps _mm128_add_ps
    #define ICAN_IRAY_SIMD_mul_ps _mm128_mul_ps
    #define ICAN_IRAY_SIMD_hadd_ps _mm128_hadd_ps
    #define ICAN_IRAY_SIMD_set1_ps _mm128_set1_ps
#endif

Iray2D *iray2d_alloc(int64 rows, int64 cols);
void iray2d_free(Iray2D *iray);
Iray2D *iray2d_add(Iray2D *A, Iray2D *B);
Iray2D *iray2d_dot(Iray2D *A, Iray2D *B);
Iray2D *iray2d_apply(Iray2D *A, float (*func)(float));
Iray2D *iray2d_scale(Iray2D *A, float scalar);
void iray2d_fill(Iray2D *A, float data);
float iray2d_max(Iray2D *A);
float iray2d_min(Iray2D *A);
Iray2D *iray2d_clone(Iray2D *A);
Iray2D *iray2d_transpose(Iray2D *A);
void iray2d_print(Iray2D *iray);

Iray1D *iray1d_alloc(int64 rows);
void iray1d_free(Iray1D *iray);
Iray1D *iray1d_add(Iray1D *A, Iray1D *B);
Iray1D *iray1d_dot(Iray1D *A, Iray1D *B);
void iray1d_fill(Iray1D *A, float data);
Iray1D *iray1d_apply(Iray1D *A, float (*func)(float));
Iray1D *iray1d_scale(Iray1D *A, float scalar);
float iray1d_max(Iray1D *A);
float iray1d_min(Iray1D *A);
Iray1D *iray1d_slice(Iray1D *A, int64 start, int64 end);
Iray1D *iray1d_clone(Iray1D *A);
void iray1d_print(Iray1D *iray);

#endif // !ICAN_IRAY_H