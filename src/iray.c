#ifndef IRAY_H

#define IRAY_H

#include "ican.h"

Iray1D *iray1d_alloc(size_t rows) {
    Iray1D *iray = malloc(sizeof(Iray1D));
    ASSERT_MSG(iray != NULL, "Failed to allocate memory");

    iray->rows = rows;
    iray->data = malloc(rows * sizeof(float));
    ASSERT_MSG(iray->data != NULL, "Failed to allocate memory for data");

    return iray;
}

void iray1d_free(Iray1D *iray) {
    if(iray != NULL) {
        free(iray->data);
        free(iray);
    }
}

Iray2D *iray2d_alloc(size_t rows, size_t cols) {
    Iray2D *iray = malloc(sizeof(Iray2D));
    ASSERT_MSG(iray != NULL, "Failed to allocate memory");

    iray->rows = rows;
    iray->cols = cols;
    iray->data = malloc(rows * sizeof(float *));
    ASSERT_MSG(iray->data != NULL, "Failed to allocate memory for data");

    for (size_t i = 0; i < rows; i++) {
        iray->data[i] = malloc(cols * sizeof(float));
        ASSERT_MSG(iray->data != NULL, "Failed to allocate memory for data");
    }
    

    return iray;
}

void iray2d_free(Iray2D *iray) {
    if(iray != NULL) {
        for (size_t i = 0; i < iray->rows; i++) {
            free(iray->data[i]);
        }
        free(iray->data);
        free(iray);
    }
}

Iray3D *iray3d_alloc(size_t rows, size_t cols, size_t depth) {
    Iray3D *iray = malloc(sizeof(Iray3D));
    ASSERT_MSG(iray != NULL, "Failed to allocate memory");

    iray->rows = rows;
    iray->cols = cols;
    iray->depth = depth;
    iray->data = malloc(rows * sizeof(float **));
    ASSERT_MSG(iray->data != NULL, "Failed to allocate memory for data");

    for (size_t i = 0; i < rows; i++) {
        iray->data[i] = malloc(cols * sizeof(float *));
        ASSERT_MSG(iray->data[i] != NULL, "Failed to allocate memory for data");

        for (size_t j = 0; j < cols; j++) {
            iray->data[i][j] = malloc(depth * sizeof(float));
            ASSERT_MSG(iray->data[i][j], "Failed to allocate memory for data");
        }
        
    }
    return iray;
}

void iray3d_free(Iray3D *iray) {
    if(iray != NULL) {
        for (size_t i = 0; i < iray->rows; i++) {
            for (size_t j = 0; j < iray->cols; j++) {
                free(iray->data[i][j]);
            }
            free(iray->data[i]);
        }
        free(iray->data);
        free(iray);
    }
}

Iray1D *iray1d_add(Iray1D *A, Iray1D *B) {
    ASSERT(A->rows == B->rows);
    Iray1D *output = iray1d_alloc(A->rows);
    for (size_t i = 0; i < A->rows; i++) {
        output->data[i] = A->data[i] + B->data[i];
    }
    return output;
}

Iray1D *iray1d_dot(Iray1D *A, Iray1D *B) {
    ASSERT(A->rows == B->rows);
    Iray1D *output = iray1d_alloc(A->rows);
    for (size_t i = 0; i < A->rows; i++) {
        output->data[i] = A->data[i] * B->data[i];
    }
    return output;
}

Iray1D *iray1d_scale(Iray1D *A, float scale) {
    Iray1D *scaled = iray1d_alloc(A->rows);
    for (size_t i = 0; i < A->rows; i++) {
        scaled->data[i] = A->data[i] / scale;
    }
    return scaled;
}

Iray1D *iray1d_slice(Iray1D *iray, size_t start, size_t end) {
    size_t new_size = end - start;
    Iray1D *result = iray1d_alloc(new_size);
    for (size_t i = start, j = 0; i < end; i++, j++) {
        result->data[j] = iray->data[i];
    }
    return result;
}

Iray1D *iray1d_apply(Iray1D *iray1d, float(*fn)(float x)) {
    Iray1D *output = iray1d_alloc(iray1d->rows);
    for (size_t i = 0; i < iray1d->rows; i++) {
        output->data[i] = fn(iray1d->data[i]);
    }
    return output;
}

Iray1D *iray1d_fill(Iray1D *iray, float value) {
    Iray1D *output = iray1d_alloc(iray->rows);
    for (size_t i = 0; i < iray->rows; i++) {
        output->data[i] = value;
    }
    return output;
}

float iray1d_max(Iray1D *iray) {
    float max = iray->data[0];
    for (size_t i = 0; i < iray->rows; i++) {
        if (iray->data[i] > max) {
            max = iray->data[i];
        }
    }
    return max;
}

Iray1D *iray1d_clone(Iray1D *iray) {
    Iray1D *output = iray1d_alloc(iray->rows);
    for (size_t i = 0; i < iray->rows; i++) {
        output->data[i] = iray->data[i];
    }
    return output;
}

void iray1d_print(Iray1D *iray) {
    printf("size = %zu\n", iray->rows);
    for (size_t i = 0; i < iray->rows; i++) {
        printf("%.1f ", iray->data[i]);
    }
}

Iray2D *iray2d_transpose(Iray2D *iray) {
    Iray2D *transposed = iray2d_alloc(iray->cols, iray->rows);
    for (size_t i = 0; i < iray->rows; i++) {
        for (size_t j = 0; j < iray->cols; j++)
        {
            transposed->data[j][i] = iray->data[i][j];
        }
    }
    return transposed;
}

Iray2D *iray2d_dot(Iray2D *A, Iray2D *B) {
    ASSERT(A->cols == B->rows);
    Iray2D *dotProduct = iray2d_alloc(A->rows, B->cols);
    for (size_t i = 0; i < A->rows; i++) {
        for (size_t j = 0; j < B->cols; j++) {
            dotProduct->data[i][j] = 0;
            for (size_t k = 0; k < A->cols; k++) {
                dotProduct->data[i][j] += A->data[i][k] * B->data[k][j];
            }
        }
    }
    return dotProduct;
}

Iray2D *iray2d_div(Iray2D *A, Iray2D *B) {
    ASSERT(A->cols == B->rows);
    Iray2D *dived = iray2d_alloc(A->rows, B->cols);
    for (size_t i = 0; i < A->rows; i++) {
        for (size_t j = 0; j < B->cols; j++) {
            dived->data[i][j] = A->data[i][j] / B->data[i][j];
        }
    }
    return dived;
}

Iray2D *iray2d_scale(Iray2D *A, float scale) {
    Iray2D *scaled = iray2d_alloc(A->rows, A->cols);
    for (size_t i = 0; i < A->rows; i++) {
        for (size_t j = 0; j < A->cols; j++) {
            scaled->data[i][j] = A->data[i][j] / scale;
        }
    }
    return scaled;
}

Iray2D *iray2d_softmax(Iray2D *matrix, int axis) {
    if (!matrix) return NULL;

    Iray2D *result = iray2d_alloc(matrix->rows, matrix->cols);

    if (axis == 0) {
        // Sütun ekseni boyunca softmax
        for (size_t j = 0; j < matrix->cols; j++) {
            float max_val = -INFINITY;
            float sum_exp = 0.0f;

            // Maksimum değeri bul
            for (size_t i = 0; i < matrix->rows; i++) {
                if (matrix->data[i][j] > max_val) {
                    max_val = matrix->data[i][j];
                }
            }

            // e^(x - max) hesapla ve toplamı al
            for (size_t i = 0; i < matrix->rows; i++) {
                result->data[i][j] = expf(matrix->data[i][j] - max_val);
                sum_exp += result->data[i][j];
            }

            // Softmax değerini hesapla
            for (size_t i = 0; i < matrix->rows; i++) {
                result->data[i][j] /= sum_exp;
            }
        }
    } else if (axis == 1) {
        // Satır ekseni boyunca softmax
        for (size_t i = 0; i < matrix->rows; i++) {
            float max_val = -INFINITY;
            float sum_exp = 0.0f;

            // Maksimum değeri bul
            for (size_t j = 0; j < matrix->cols; j++) {
                if (matrix->data[i][j] > max_val) {
                    max_val = matrix->data[i][j];
                }
            }

            // e^(x - max) hesapla ve toplamı al
            for (size_t j = 0; j < matrix->cols; j++) {
                result->data[i][j] = expf(matrix->data[i][j] - max_val);
                sum_exp += result->data[i][j];
            }

            // Softmax değerini hesapla
            for (size_t j = 0; j < matrix->cols; j++) {
                result->data[i][j] /= sum_exp;
            }
        }
    }

    return result;
}

Iray2D *iray2d_concat(Iray2D **matrices, size_t num_matrices) {
  if (num_matrices == 0) return NULL;

  size_t rows = matrices[0]->rows;

  size_t total_cols = 0;
  for (size_t i = 0; i < num_matrices; i++) {
    total_cols += matrices[i]->cols;
  }

  Iray2D *concat = iray2d_alloc(rows, total_cols);

  size_t col_offset = 0;
  for (size_t i = 0; i < num_matrices; i++) {
    for (size_t r = 0; r < rows; r++) {
      for (size_t c = 0; c < matrices[i]->cols; c++) {
        concat->data[r][col_offset + c] = matrices[i]->data[r][c];
      }
    }
    col_offset += matrices[i]->cols;
  }

  return concat;
}

Iray2D *iray2d_slice(Iray2D *iray, size_t start, size_t end) {
    size_t new_size = end - start;
    Iray2D *result = iray2d_alloc(new_size, iray->cols);
    for (size_t i = start, j = 0; i < end; i++, j++) {
        for (size_t k = 0; k < iray->cols; k++) {
            result->data[j][k] = iray->data[i][k];
        }
    }
    return result;
}

Iray2D *iray2d_add(Iray2D *A, Iray2D *B) {
    ASSERT(A->rows == B->rows);
    ASSERT(A->cols == B->cols);
    Iray2D *sum = iray2d_alloc(A->rows, B->cols);
    for (size_t i = 0; i < A->rows; i++) {
        for (size_t j = 0; j < A->cols; j++) {
            sum->data[i][j] += A->data[i][j] + B->data[i][j];
        }
    }
    return sum;
}

Iray2D *iray2d_apply(Iray2D *iray, float(*fn)(float value)) {
    Iray2D *applied = iray2d_alloc(iray->rows, iray->cols);
    for (size_t i = 0; i < iray->rows; i++) {
        for (size_t j = 0; j < iray->cols; j++) {
            applied->data[i][j] = fn(iray->data[i][j]);
        }
    }
    return applied;
}

Iray2D *iray2d_fill(Iray2D *iray, float value) {
    Iray2D *filled = iray2d_alloc(iray->rows, iray->cols);
    for (size_t i = 0; i < iray->rows; i++) {
        for (size_t j = 0; j < iray->cols; j++) {
            filled->data[i][j] = value;
        }
    }
    return filled;
}

float iray2d_max(Iray2D *iray) {
    float max = iray->data[0][0];
    for (size_t i = 0; i < iray->rows; i++) {
        for (size_t j = 0; j < iray->cols; j++) {
            if (iray->data[i][j] > max) {
                max = iray->data[i][j];
            }
        }
    }
    return max;
}

Iray2D *iray2d_clone(Iray2D *iray) {
    Iray2D *output = iray2d_alloc(iray->rows, iray->cols);
    for (size_t i = 0; i < iray->rows; i++) {
        for (size_t j = 0; j < iray->cols; j++) {
            output->data[i][j] = iray->data[i][j];
        }
        
    }
    return output;
}

void iray2d_print(Iray2D *iray) {
     printf("rows = %zu, cols = %zu\n", iray->rows, iray->cols);
    for (size_t i = 0; i < iray->rows; i++) {
        printf("[");
        for (size_t j = 0; j < iray->cols; j++) {
            printf(" %.3f", iray->data[i][j]);
        }
        printf(" ]\n");
    }
    printf("\n");
}

Iray3D *iray3d_add(Iray3D *A, Iray3D *B) {
    ASSERT(A->rows == B->rows);
    ASSERT(A->cols == B->cols);
    ASSERT(A->depth == B->depth);
    
    Iray3D *sum = iray3d_alloc(A->rows, A->cols, A->depth);
    for (size_t i = 0; i < A->rows; i++) {
        for (size_t j = 0; j < A->cols; j++) {
            for (size_t k = 0; k < A->depth; k++) {
                sum->data[i][j][k] = A->data[i][j][k] + B->data[i][j][k];
            }
        }
    }
    return sum;
}

Iray3D *iray3d_apply(Iray3D *iray, float (*fn)(float value)) {
    Iray3D *applied = iray3d_alloc(iray->rows, iray->cols, iray->depth);
    for (size_t i = 0; i < iray->rows; i++) {
        for (size_t j = 0; j < iray->cols; j++) {
            for (size_t k = 0; k < iray->depth; k++) {
                applied->data[i][j][k] = fn(iray->data[i][j][k]);
            }
        }
    }
    return applied;
}

Iray3D *iray3d_fill(Iray3D *iray, float value) {
    Iray3D *filled = iray3d_alloc(iray->rows, iray->cols, iray->depth);
    for (size_t i = 0; i < iray->rows; i++) {
        for (size_t j = 0; j < iray->cols; j++) {
            for (size_t k = 0; k < iray->depth; k++) {
                filled->data[i][j][k] = value;
            }
        }
    }
    return filled;
}

float iray3d_max(Iray3D *iray) {
    float max = iray->data[0][0][0];
    for (size_t i = 0; i < iray->rows; i++) {
        for (size_t j = 0; j < iray->cols; j++) {
            for (size_t k = 0; k < iray->depth; k++) {
                if(iray->data[i][j][k] > max) {
                    max = iray->data[i][j][k];
                }
            }
        }
    }
    return max;
}

Iray3D *iray3d_clone(Iray3D *iray) {
    Iray3D *clone = iray3d_alloc(iray->rows, iray->cols, iray->depth);
    for (size_t i = 0; i < iray->rows; i++) {
        for (size_t j = 0; j < iray->cols; j++) {
            for (size_t k = 0; k < iray->depth; k++) {
                clone->data[i][j][k] = iray->data[i][j][k];
            }
        }
    }
    return clone;
}

void iray3d_print(Iray3D *iray) {
    printf("rows = %zu, cols = %zu, depth = %zu\n", iray->rows, iray->cols, iray->depth);
    for (size_t i = 0; i < iray->rows; i++) {
        printf("Layer %zu:\n", i);
        for (size_t j = 0; j < iray->cols; j++) {
            printf("[");
            for (size_t k = 0; k < iray->depth; k++) {
                printf(" %.3f", iray->data[i][j][k]);
            }
            printf(" ]\n");
        }
        printf("\n");
    }
}

Iray3D *iray3d_transpose(Iray3D *iray) {
    Iray3D *transposed = iray3d_alloc(iray->depth, iray->rows, iray->cols);
    for (size_t i = 0; i < iray->rows; i++) {
        for (size_t j = 0; j < iray->cols; j++) {
            for (size_t k = 0; k < iray->depth; k++) {
                transposed->data[k][i][j] = iray->data[i][j][k];
            }
        }
    }
    return transposed;
}

#endif // IRAY_H