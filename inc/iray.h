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