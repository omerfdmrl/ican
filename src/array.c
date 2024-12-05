#include "data_structure/array.h"

Array2D *array2d_alloc(int64 rows, int64 cols) {
    Array2D *array = (Array2D *) ICAN_MALLOC(sizeof(Array2D));
    ASSERT(array != NULL);
    array->rows = rows;
    array->cols = cols;
    array->data = (float *) aligned_alloc(ICAN_ARRAY_SIMD_align, sizeof(float) * rows * cols);
    ASSERT(array->data != NULL);
    return array;
}

void array2d_free(Array2D *array) {
    if(array != NULL) {
        if (array->data != NULL) {
            ICAN_FREE(array->data);
        }
        ICAN_FREE(array);
    }
}

Array2D *array2d_add(Array2D *A, Array2D *B) {
    ASSERT(A->cols == B->cols);
    ASSERT(A->rows == B->rows);
    Array2D *O = array2d_alloc(A->rows, A->cols);
    
    #pragma omp parallel for
    for (size_t i = 0; i < A->rows; i++) {
        for (size_t j = 0; j < A->cols; j += ICAN_ARRAY_SIMD) {
            size_t vec_end = (j + ICAN_ARRAY_SIMD > A->cols) ? A->cols : j + ICAN_ARRAY_SIMD;
            if (vec_end - j == ICAN_ARRAY_SIMD) {
                __m256 vec_a = ICAN_ARRAY_SIMD_load_ps(&A->data[i * A->cols + j]);
                __m256 vec_b = ICAN_ARRAY_SIMD_load_ps(&B->data[i * B->cols + j]);
                __m256 vec_sum = ICAN_ARRAY_SIMD_add_ps(vec_a, vec_b);
                ICAN_ARRAY_SIMD_store_ps(&O->data[i * O->cols + j], vec_sum);
            } else {
                for (size_t k = j; k < vec_end; k++) {
                    O->data[i * O->cols + k] = A->data[i * A->cols + k] + B->data[i * B->cols + k];
                }
            }
        }
    }
    return O;
}

void array2d_add_inplace(Array2D *A, Array2D *B) {
    ASSERT(A->cols == B->cols);
    ASSERT(A->rows == B->rows);
    
    #pragma omp parallel for
    for (size_t i = 0; i < A->rows; i++) {
        for (size_t j = 0; j < A->cols; j += ICAN_ARRAY_SIMD) {
            size_t vec_end = (j + ICAN_ARRAY_SIMD > A->cols) ? A->cols : j + ICAN_ARRAY_SIMD;
            if (vec_end - j == ICAN_ARRAY_SIMD) {
                __m256 vec_a = ICAN_ARRAY_SIMD_load_ps(&A->data[i * A->cols + j]);
                __m256 vec_b = ICAN_ARRAY_SIMD_load_ps(&B->data[i * B->cols + j]);
                __m256 vec_sum = ICAN_ARRAY_SIMD_add_ps(vec_a, vec_b);
                ICAN_ARRAY_SIMD_store_ps(&A->data[i * A->cols + j], vec_sum);
            } else {
                for (size_t k = j; k < vec_end; k++) {
                    A->data[i * A->cols + k] = A->data[i * A->cols + k] + B->data[i * B->cols + k];
                }
            }
        }
    }
}

Array2D *array2d_sub(Array2D *A, Array2D *B) {
    ASSERT(A->cols == B->cols);
    ASSERT(A->rows == B->rows);
    Array2D *O = array2d_alloc(A->rows, A->cols);
    
    #pragma omp parallel for
    for (size_t i = 0; i < A->rows; i++) {
        for (size_t j = 0; j < A->cols; j += ICAN_ARRAY_SIMD) {
            size_t vec_end = (j + ICAN_ARRAY_SIMD > A->cols) ? A->cols : j + ICAN_ARRAY_SIMD;
            if (vec_end - j == ICAN_ARRAY_SIMD) {
                __m256 vec_a = ICAN_ARRAY_SIMD_load_ps(&A->data[i * A->cols + j]);
                __m256 vec_b = ICAN_ARRAY_SIMD_load_ps(&B->data[i * B->cols + j]);
                __m256 vec_sum = ICAN_ARRAY_SIMD_sub_ps(vec_a, vec_b);
                ICAN_ARRAY_SIMD_store_ps(&O->data[i * O->cols + j], vec_sum);
            } else {
                for (size_t k = j; k < vec_end; k++) {
                    O->data[i * O->cols + k] = A->data[i * A->cols + k] + B->data[i * B->cols + k];
                }
            }
        }
    }
    return O;
}

void array2d_sub_inplace(Array2D *A, Array2D *B) {
    ASSERT(A->cols == B->cols);
    ASSERT(A->rows == B->rows);
    
    #pragma omp parallel for
    for (size_t i = 0; i < A->rows; i++) {
        for (size_t j = 0; j < A->cols; j += ICAN_ARRAY_SIMD) {
            size_t vec_end = (j + ICAN_ARRAY_SIMD > A->cols) ? A->cols : j + ICAN_ARRAY_SIMD;
            if (vec_end - j == ICAN_ARRAY_SIMD) {
                __m256 vec_a = ICAN_ARRAY_SIMD_load_ps(&A->data[i * A->cols + j]);
                __m256 vec_b = ICAN_ARRAY_SIMD_load_ps(&B->data[i * B->cols + j]);
                __m256 vec_sum = ICAN_ARRAY_SIMD_sub_ps(vec_a, vec_b);
                ICAN_ARRAY_SIMD_store_ps(&A->data[i * A->cols + j], vec_sum);
            } else {
                for (size_t k = j; k < vec_end; k++) {
                    A->data[i * A->cols + k] = A->data[i * A->cols + k] + B->data[i * B->cols + k];
                }
            }
        }
    }
}

Array2D *array2d_dot(Array2D *A, Array2D *B) {
    ASSERT(A->cols == B->rows);
    Array2D *O = array2d_alloc(A->rows, B->cols);

    #pragma omp parallel for
    for (size_t i = 0; i < A->rows; i++) {
        for (size_t j = 0; j < B->cols; j++) {
            O->data[i * O->cols + j] = 0.0f;
            for (size_t k = 0; k < A->cols; k+=ICAN_ARRAY_SIMD) {
                size_t vec_end = (k + ICAN_ARRAY_SIMD > A->cols) ? A->cols : k + ICAN_ARRAY_SIMD;
                if (vec_end - k == ICAN_ARRAY_SIMD) {
                    __m256 vec_a = ICAN_ARRAY_SIMD_load_ps(&A->data[i * A->cols + k]);
                    __m256 vec_b = ICAN_ARRAY_SIMD_load_ps(&B->data[k * B->cols + j]);
                    __m256 vec_p = ICAN_ARRAY_SIMD_mul_ps(vec_a, vec_b);
                    __m256 vec_s = ICAN_ARRAY_SIMD_hadd_ps(vec_p, vec_p);
                    vec_s = ICAN_ARRAY_SIMD_hadd_ps(vec_s, vec_s);
                    float result[ICAN_ARRAY_SIMD];
                    ICAN_ARRAY_SIMD_store_ps(result, vec_s);
                    O->data[i * O->cols + j] += result[0] + result[4];
                } else {
                    for (size_t l = k; l < vec_end; l++) {
                        O->data[i * O->cols + j] += A->data[i * A->cols + l] * B->data[l * B->cols + j];
                    }
                }
            }
        }
    }

    return O;
}

void array2d_dot_inplace(Array2D *A, Array2D *B) {
    ASSERT(A->cols == B->rows);

    #pragma omp parallel for
    for (size_t i = 0; i < A->rows; i++) {
        for (size_t j = 0; j < B->cols; j++) {
            A->data[i * A->cols + j] = 0.0f;
            for (size_t k = 0; k < A->cols; k+=ICAN_ARRAY_SIMD) {
                size_t vec_end = (k + ICAN_ARRAY_SIMD > A->cols) ? A->cols : k + ICAN_ARRAY_SIMD;
                if (vec_end - k == ICAN_ARRAY_SIMD) {
                    __m256 vec_a = ICAN_ARRAY_SIMD_load_ps(&A->data[i * A->cols + k]);
                    __m256 vec_b = ICAN_ARRAY_SIMD_load_ps(&B->data[k * B->cols + j]);
                    __m256 vec_p = ICAN_ARRAY_SIMD_mul_ps(vec_a, vec_b);
                    __m256 vec_s = ICAN_ARRAY_SIMD_hadd_ps(vec_p, vec_p);
                    vec_s = ICAN_ARRAY_SIMD_hadd_ps(vec_s, vec_s);
                    float result[ICAN_ARRAY_SIMD];
                    ICAN_ARRAY_SIMD_store_ps(result, vec_s);
                    A->data[i * A->cols + j] += result[0] + result[4];
                } else {
                    for (size_t l = k; l < vec_end; l++) {
                        A->data[i * A->cols + j] += A->data[i * A->cols + l] * B->data[l * B->cols + j];
                    }
                }
            }
        }
    }
}

Array2D *array2d_apply(Array2D *A, float (*func)(float)) {
    Array2D *O = array2d_alloc(A->rows, A->cols);
    
    #pragma omp parallel for
    for (size_t i = 0; i < A->rows; i++) {
        for (size_t j = 0; j < A->cols; j += ICAN_ARRAY_SIMD) {
            size_t vec_end = (j + ICAN_ARRAY_SIMD > A->cols) ? A->cols : j + ICAN_ARRAY_SIMD;
            if (vec_end - j == ICAN_ARRAY_SIMD) {
                __m256 vec_a = ICAN_ARRAY_SIMD_load_ps(&A->data[i * A->cols + j]);
                for (size_t k = 0; k < ICAN_ARRAY_SIMD; k++) {
                    float val = func(((float*)&vec_a)[k]);
                    ((float*)&vec_a)[k] = val;
                }
                ICAN_ARRAY_SIMD_store_ps(&O->data[i * O->cols + j], vec_a);
            } else {
                for (size_t k = j; k < vec_end; k++) {
                    O->data[i * O->cols + k] = func(A->data[i * A->cols + k]);
                }
            }
        }
    }
    return O;
}

void array2d_apply_inplace(Array2D *A, float (*func)(float)) {
    #pragma omp parallel for
    for (size_t i = 0; i < A->rows; i++) {
        for (size_t j = 0; j < A->cols; j += ICAN_ARRAY_SIMD) {
            size_t vec_end = (j + ICAN_ARRAY_SIMD > A->cols) ? A->cols : j + ICAN_ARRAY_SIMD;
            if (vec_end - j == ICAN_ARRAY_SIMD) {
                __m256 vec_a = ICAN_ARRAY_SIMD_load_ps(&A->data[i * A->cols + j]);
                for (size_t k = 0; k < ICAN_ARRAY_SIMD; k++) {
                    float val = func(((float*)&vec_a)[k]);
                    ((float*)&vec_a)[k] = val;
                }
                ICAN_ARRAY_SIMD_store_ps(&A->data[i * A->cols + j], vec_a);
            } else {
                for (size_t k = j; k < vec_end; k++) {
                    A->data[i * A->cols + k] = func(A->data[i * A->cols + k]);
                }
            }
        }
    }
}

Array2D *array2d_sort(Array2D *A, int (*comp)(float*, float*, int64)) {
    Array2D *O = array2d_alloc(A->rows, A->cols);
    memcpy(O->data, A->data, A->rows * A->cols * sizeof(float));
    bool sorted = false;
    while (!sorted) {
        sorted = true;
        for (size_t i = 0; i < O->rows - 1; i++) {
            if (comp(&O->data[i * O->cols], &O->data[(i + 1) * O->cols], O->cols) > 0) {
                Array1D *temp = array1d_from(&O->data[i * O->cols], O->cols);
                memcpy(&O->data[i * O->cols], &O->data[(i + 1) * O->cols], O->cols * sizeof(float));
                memcpy(&O->data[(i + 1) * O->cols], temp->data, O->cols * sizeof(float));
                array1d_free(temp);
                sorted = false;
            }
        }
    }
    return O;
}

void array2d_sort_inplace(Array2D *A, int (*comp)(float*, float*, int64)) {
    bool sorted = false;
    while (!sorted) {
        sorted = true;
        for (size_t i = 0; i < A->rows - 1; i++) {
            if (comp(&A->data[i * A->cols], &A->data[(i + 1) * A->cols], A->cols) > 0) {
                Array1D *temp = array1d_from(&A->data[i * A->cols], A->cols);
                memcpy(&A->data[i * A->cols], &A->data[(i + 1) * A->cols], A->cols * sizeof(float));
                memcpy(&A->data[(i + 1) * A->cols], temp->data, A->cols * sizeof(float));
                array1d_free(temp);
                sorted = false;
            }
        }
    }
}

Array2D *array2d_mul_scalar(Array2D *A, float scalar) {
    __m256 vec_scalar = ICAN_ARRAY_SIMD_set1_ps(scalar);
    Array2D *O = array2d_alloc(A->rows, A->cols);
    
    #pragma omp parallel for
    for (size_t i = 0; i < A->rows; i++) {
        for (size_t j = 0; j < A->cols; j += ICAN_ARRAY_SIMD) {
            size_t vec_end = (j + ICAN_ARRAY_SIMD > A->cols) ? A->cols : j + ICAN_ARRAY_SIMD;
            if (vec_end - j == ICAN_ARRAY_SIMD) {
                __m256 vec_a = ICAN_ARRAY_SIMD_load_ps(&A->data[i * A->cols + j]);
                __m256 vec_o = ICAN_ARRAY_SIMD_mul_ps(vec_a, vec_scalar);
                ICAN_ARRAY_SIMD_store_ps(&O->data[i * O->cols + j], vec_o);
            } else {
                for (size_t k = j; k < vec_end; k++) {
                    O->data[i * O->cols + k] = A->data[i * A->cols + k] * scalar;
                }
            }
        }
    }
    return O;
}

void array2d_mul_scalar_inplace(Array2D *A, float scalar) {
    __m256 vec_scalar = ICAN_ARRAY_SIMD_set1_ps(scalar);
    
    #pragma omp parallel for
    for (size_t i = 0; i < A->rows; i++) {
        for (size_t j = 0; j < A->cols; j += ICAN_ARRAY_SIMD) {
            size_t vec_end = (j + ICAN_ARRAY_SIMD > A->cols) ? A->cols : j + ICAN_ARRAY_SIMD;
            if (vec_end - j == ICAN_ARRAY_SIMD) {
                __m256 vec_a = ICAN_ARRAY_SIMD_load_ps(&A->data[i * A->cols + j]);
                __m256 vec_o = ICAN_ARRAY_SIMD_mul_ps(vec_a, vec_scalar);
                ICAN_ARRAY_SIMD_store_ps(&A->data[i * A->cols + j], vec_o);
            } else {
                for (size_t k = j; k < vec_end; k++) {
                    A->data[i * A->cols + k] = A->data[i * A->cols + k] * scalar;
                }
            }
        }
    }
}

Array2D *array2d_fill(Array2D *A, float data) {
    __m256 vec_data = ICAN_ARRAY_SIMD_set1_ps(data);
    Array2D *O = array2d_alloc(A->rows, A->cols);

    #pragma omp parallel for
    for (size_t i = 0; i < A->rows; i++) {
        for (size_t j = 0; j < A->cols; j += ICAN_ARRAY_SIMD) {
            size_t vec_end = (j + ICAN_ARRAY_SIMD > A->cols) ? A->cols : j + ICAN_ARRAY_SIMD;
            if (vec_end - j == ICAN_ARRAY_SIMD) {
                ICAN_ARRAY_SIMD_store_ps(&O->data[i * A->cols + j], vec_data);
            } else {
                for (size_t k = j; k < vec_end; k++) {
                    O->data[i * A->cols + k] = data;
                }
            }
        }
    }
    return O;
}

void array2d_fill_inplace(Array2D *A, float data) {
    __m256 vec_data = ICAN_ARRAY_SIMD_set1_ps(data);
    
    #pragma omp parallel for
    for (size_t i = 0; i < A->rows; i++) {
        for (size_t j = 0; j < A->cols; j += ICAN_ARRAY_SIMD) {
            size_t vec_end = (j + ICAN_ARRAY_SIMD > A->cols) ? A->cols : j + ICAN_ARRAY_SIMD;
            if (vec_end - j == ICAN_ARRAY_SIMD) {
                ICAN_ARRAY_SIMD_store_ps(&A->data[i * A->cols + j], vec_data);
            } else {
                for (size_t k = j; k < vec_end; k++) {
                    A->data[i * A->cols + k] = data;
                }
            }
        }
    }
}

float array2d_max(Array2D *A) {
    float max_value = A->data[0];
    #pragma omp parallel for reduction(max:max_value)
    for (size_t i = 0; i < A->rows; i++) {
        for (size_t j = 0; j < A->cols; j += ICAN_ARRAY_SIMD) {
            size_t vec_end = (j + ICAN_ARRAY_SIMD > A->cols) ? A->cols : j + ICAN_ARRAY_SIMD;
            if (vec_end - j == ICAN_ARRAY_SIMD) {
                __m256 vec_a = ICAN_ARRAY_SIMD_load_ps(&A->data[i * A->cols + j]);
                for (size_t k = 0; k < ICAN_ARRAY_SIMD; k++) {
                    max_value = fmaxf(max_value, ((float*)&vec_a)[k]);
                }
            } else {
                for (size_t k = j; k < vec_end; k++) {
                    max_value = fmaxf(max_value, A->data[i * A->cols + k]);
                }
            }
        }
    }
    return max_value;
}

float array2d_min(Array2D *A) {
    float min_value = A->data[0];
    #pragma omp parallel for reduction(min:min_value)
    for (size_t i = 0; i < A->rows; i++) {
        for (size_t j = 0; j < A->cols; j += ICAN_ARRAY_SIMD) {
            size_t vec_end = (j + ICAN_ARRAY_SIMD > A->cols) ? A->cols : j + ICAN_ARRAY_SIMD;
            if (vec_end - j == ICAN_ARRAY_SIMD) {
                __m256 vec_a = ICAN_ARRAY_SIMD_load_ps(&A->data[i * A->cols + j]);
                for (size_t k = 0; k < ICAN_ARRAY_SIMD; k++) {
                    min_value = fminf(min_value, ((float*)&vec_a)[k]);
                }
            } else {
                for (size_t k = j; k < vec_end; k++) {
                    min_value = fminf(min_value, A->data[i * A->cols + k]);
                }
            }
        }
    }
    return min_value;
}

Array2D *array2d_slice(Array2D *A, int64 row_start, int64 row_end, int64 col_start, int64 col_end) {
    ASSERT(row_start >= 0 && row_start < A->rows);
    ASSERT(row_end > row_start && row_end <= A->rows);
    ASSERT(col_start >= 0 && col_start < A->cols);
    ASSERT(col_end > col_start && col_end <= A->cols);
    int64 new_rows = row_end - row_start;
    int64 new_cols = col_end - col_start;
    Array2D *O = array2d_alloc(new_rows, new_cols);
    #pragma omp parallel for
    for (int i = 0; i < new_rows; i++) {
        memcpy(
            &O->data[i * new_cols],
            &A->data[(row_start + i) * A->cols + col_start],
            sizeof(float) * new_cols
        );
    }
    return O;
}

Array2D *array2d_clone(Array2D *A) {
    Array2D *O = array2d_alloc(A->rows, A->cols);
    memcpy(O->data, A->data, A->rows * A->cols * sizeof(float));
    return O;
}

Array2D *array2d_transpose(Array2D *A) {
    Array2D *O = array2d_alloc(A->cols, A->rows);
    #pragma omp parallel for
    for (size_t i = 0; i < A->rows; i++) {
        for (size_t j = 0; j < A->cols; j ++) {
            O->data[j * O->cols + i] = A->data[i * A->cols + j];
        }
    }
    return O;
}

Array2D *array2d_from(float *data, int64 rows, int64 cols) {
    Array2D *O = array2d_alloc(rows, cols);
    memcpy(O->data, data, rows * cols * sizeof(float));
    return O;
}

Array1D *array2d_to_array1d(Array2D *A) {
    Array1D *O = array1d_from(A->data, A->rows * A->cols);
    return O;
}

void array2d_print(Array2D *array) {
    printf("Rows = %lld, Cols = %lld\n", array->rows, array->cols);
    for (int i = 0; i < array->rows; i++) {
        printf("[");
        for (int j = 0; j < array->cols; j++) {
            printf("%f", array->data[i * array->cols + j]);
            if(j < array->cols) {
                printf(" ");
            }
        }
        printf("]\n");
    }
}

Array1D *array1d_alloc(int64 rows) {
    Array1D *array = (Array1D *) ICAN_MALLOC(sizeof(Array1D));
    ASSERT(array != NULL);
    array->rows = rows;
    array->data = (float *) aligned_alloc(ICAN_ARRAY_SIMD_align, sizeof(float) * rows);
    ASSERT(array->data != NULL);
    return array;
}

void array1d_free(Array1D *array) {
    if(array != NULL) {
        if (array->data != NULL) {
            ICAN_FREE(array->data);
        }
        ICAN_FREE(array);
    }
}

Array1D *array1d_add(Array1D *A, Array1D *B) {
    ASSERT(A->rows == B->rows);
    Array1D *O = array1d_alloc(A->rows);

    #pragma omp parallel for
    for (size_t i = 0; i < A->rows; i+=ICAN_ARRAY_SIMD) {
        size_t vec_end = (i + ICAN_ARRAY_SIMD > A->rows) ? A->rows : i + ICAN_ARRAY_SIMD;
        if (vec_end - i == ICAN_ARRAY_SIMD) {
            __m256 vec_a = ICAN_ARRAY_SIMD_load_ps(&A->data[i]);
            __m256 vec_b = ICAN_ARRAY_SIMD_load_ps(&B->data[i]);
            __m256 vec_sum = ICAN_ARRAY_SIMD_add_ps(vec_a, vec_b);
            ICAN_ARRAY_SIMD_store_ps(&O->data[i], vec_sum);
        } else {
            for (size_t k = i; k < vec_end; k++) {
                O->data[k] = A->data[k] + B->data[k];
            }
        }
    }
    return O;
}

void array1d_add_inplace(Array1D *A, Array1D *B) {
    ASSERT(A->rows == B->rows);

    #pragma omp parallel for
    for (size_t i = 0; i < A->rows; i+=ICAN_ARRAY_SIMD) {
        size_t vec_end = (i + ICAN_ARRAY_SIMD > A->rows) ? A->rows : i + ICAN_ARRAY_SIMD;
        if (vec_end - i == ICAN_ARRAY_SIMD) {
            __m256 vec_a = ICAN_ARRAY_SIMD_load_ps(&A->data[i]);
            __m256 vec_b = ICAN_ARRAY_SIMD_load_ps(&B->data[i]);
            __m256 vec_sub = ICAN_ARRAY_SIMD_add_ps(vec_a, vec_b);
            ICAN_ARRAY_SIMD_store_ps(&A->data[i], vec_sub);
        } else {
            for (size_t k = i; k < vec_end; k++) {
                A->data[k] += B->data[k];
            }
        }
    }
}

Array1D *array1d_add_scalar(Array1D *A, float data) {
    Array1D *O = array1d_alloc(A->rows);
    __m256 vec_b = ICAN_ARRAY_SIMD_set1_ps(data);
    #pragma omp parallel for
    for (size_t i = 0; i < A->rows; i+=ICAN_ARRAY_SIMD) {
        size_t vec_end = (i + ICAN_ARRAY_SIMD > A->rows) ? A->rows : i + ICAN_ARRAY_SIMD;
        if (vec_end - i == ICAN_ARRAY_SIMD) {
            __m256 vec_a = ICAN_ARRAY_SIMD_load_ps(&A->data[i]);
            __m256 vec_sub = ICAN_ARRAY_SIMD_add_ps(vec_a, vec_b);
            ICAN_ARRAY_SIMD_store_ps(&O->data[i], vec_sub);
        } else {
            for (size_t k = i; k < vec_end; k++) {
                O->data[k] = A->data[k] + data;
            }
        }
    }
    return O;
}

void array1d_add_scalar_inplace(Array1D *A, float data) {
    __m256 vec_b = ICAN_ARRAY_SIMD_set1_ps(data);
    #pragma omp parallel for
    for (size_t i = 0; i < A->rows; i+=ICAN_ARRAY_SIMD) {
        size_t vec_end = (i + ICAN_ARRAY_SIMD > A->rows) ? A->rows : i + ICAN_ARRAY_SIMD;
        if (vec_end - i == ICAN_ARRAY_SIMD) {
            __m256 vec_a = ICAN_ARRAY_SIMD_load_ps(&A->data[i]);
            __m256 vec_sub = ICAN_ARRAY_SIMD_add_ps(vec_a, vec_b);
            ICAN_ARRAY_SIMD_store_ps(&A->data[i], vec_sub);
        } else {
            for (size_t k = i; k < vec_end; k++) {
                A->data[k] = A->data[k] + data;
            }
        }
    }
}

Array1D *array1d_sub(Array1D *A, Array1D *B) {
    ASSERT(A->rows == B->rows);
    Array1D *O = array1d_alloc(A->rows);

    #pragma omp parallel for
    for (size_t i = 0; i < A->rows; i+=ICAN_ARRAY_SIMD) {
        size_t vec_end = (i + ICAN_ARRAY_SIMD > A->rows) ? A->rows : i + ICAN_ARRAY_SIMD;
        if (vec_end - i == ICAN_ARRAY_SIMD) {
            __m256 vec_a = ICAN_ARRAY_SIMD_load_ps(&A->data[i]);
            __m256 vec_b = ICAN_ARRAY_SIMD_load_ps(&B->data[i]);
            __m256 vec_sub = ICAN_ARRAY_SIMD_sub_ps(vec_a, vec_b);
            ICAN_ARRAY_SIMD_store_ps(&O->data[i], vec_sub);
        } else {
            for (size_t k = i; k < vec_end; k++) {
                O->data[k] = A->data[k] - B->data[k];
            }
        }
    }
    return O;
}

void array1d_sub_inplace(Array1D *A, Array1D *B) {
    ASSERT(A->rows == B->rows);

    #pragma omp parallel for
    for (size_t i = 0; i < A->rows; i+=ICAN_ARRAY_SIMD) {
        size_t vec_end = (i + ICAN_ARRAY_SIMD > A->rows) ? A->rows : i + ICAN_ARRAY_SIMD;
        if (vec_end - i == ICAN_ARRAY_SIMD) {
            __m256 vec_a = ICAN_ARRAY_SIMD_load_ps(&A->data[i]);
            __m256 vec_b = ICAN_ARRAY_SIMD_load_ps(&B->data[i]);
            __m256 vec_sub = ICAN_ARRAY_SIMD_sub_ps(vec_a, vec_b);
            ICAN_ARRAY_SIMD_store_ps(&A->data[i], vec_sub);
        } else {
            for (size_t k = i; k < vec_end; k++) {
                A->data[k] -= B->data[k];
            }
        }
    }
}

Array1D *array1d_sub_scalar(Array1D *A, float data) {
    Array1D *O = array1d_alloc(A->rows);
    __m256 vec_b = ICAN_ARRAY_SIMD_set1_ps(data);
    #pragma omp parallel for
    for (size_t i = 0; i < A->rows; i+=ICAN_ARRAY_SIMD) {
        size_t vec_end = (i + ICAN_ARRAY_SIMD > A->rows) ? A->rows : i + ICAN_ARRAY_SIMD;
        if (vec_end - i == ICAN_ARRAY_SIMD) {
            __m256 vec_a = ICAN_ARRAY_SIMD_load_ps(&A->data[i]);
            __m256 vec_sub = ICAN_ARRAY_SIMD_sub_ps(vec_a, vec_b);
            ICAN_ARRAY_SIMD_store_ps(&O->data[i], vec_sub);
        } else {
            for (size_t k = i; k < vec_end; k++) {
                O->data[k] = A->data[k] - data;
            }
        }
    }
    return O;
}

void array1d_sub_scalar_inplace(Array1D *A, float data) {
    __m256 vec_b = ICAN_ARRAY_SIMD_set1_ps(data);
    #pragma omp parallel for
    for (size_t i = 0; i < A->rows; i+=ICAN_ARRAY_SIMD) {
        size_t vec_end = (i + ICAN_ARRAY_SIMD > A->rows) ? A->rows : i + ICAN_ARRAY_SIMD;
        if (vec_end - i == ICAN_ARRAY_SIMD) {
            __m256 vec_a = ICAN_ARRAY_SIMD_load_ps(&A->data[i]);
            __m256 vec_sub = ICAN_ARRAY_SIMD_sub_ps(vec_a, vec_b);
            ICAN_ARRAY_SIMD_store_ps(&A->data[i], vec_sub);
        } else {
            for (size_t k = i; k < vec_end; k++) {
                A->data[k] = A->data[k] - data;
            }
        }
    }
}

float array1d_sum(Array1D *A) {
    float total_sum = 0.0f;

    #pragma omp parallel
    {
        __m256 vec_sum = ICAN_ARRAY_SIMD_setzero_ps();
        float thread_sum = 0.0f;

        #pragma omp for
        for (size_t i = 0; i < A->rows; i += 8) {
            if (i + 8 <= A->rows) {
                __m256 vec_a = ICAN_ARRAY_SIMD_load_ps(&A->data[i]);
                vec_sum = ICAN_ARRAY_SIMD_add_ps(vec_sum, vec_a);
            } else {
                for (size_t j = i; j < A->rows; ++j) {
                    thread_sum += A->data[j];
                }
            }
        }

        float vec_total[8];
        ICAN_ARRAY_SIMD_store_ps(vec_total, vec_sum);
        for (int k = 0; k < 8; ++k) {
            thread_sum += vec_total[k];
        }

        #pragma omp atomic
        total_sum += thread_sum;
    }
    return total_sum;
}

Array1D *array1d_dot(Array1D *A, Array1D *B) {
    ASSERT(A->rows == B->rows);
    Array1D *O = array1d_alloc(A->rows);

    #pragma omp parallel for
    for (size_t i = 0; i < A->rows; i+=ICAN_ARRAY_SIMD) {
        size_t vec_end = (i + ICAN_ARRAY_SIMD > A->rows) ? A->rows : i + ICAN_ARRAY_SIMD;
        if (vec_end - i == ICAN_ARRAY_SIMD) {
            __m256 vec_a = ICAN_ARRAY_SIMD_load_ps(&A->data[i]);
            __m256 vec_b = ICAN_ARRAY_SIMD_load_ps(&B->data[i]);
            __m256 vec_p = ICAN_ARRAY_SIMD_mul_ps(vec_a, vec_b);
            ICAN_ARRAY_SIMD_store_ps(&O->data[i], vec_p);
        } else {
            for (size_t k = i; k < vec_end; k++) {
                O->data[k] = A->data[k] * B->data[k];
            }
        }
    }
    return O;
}

void array1d_dot_inplace(Array1D *A, Array1D *B) {
    ASSERT(A->rows == B->rows);

    #pragma omp parallel for
    for (size_t i = 0; i < A->rows; i+=ICAN_ARRAY_SIMD) {
        size_t vec_end = (i + ICAN_ARRAY_SIMD > A->rows) ? A->rows : i + ICAN_ARRAY_SIMD;
        if (vec_end - i == ICAN_ARRAY_SIMD) {
            __m256 vec_a = ICAN_ARRAY_SIMD_load_ps(&A->data[i]);
            __m256 vec_b = ICAN_ARRAY_SIMD_load_ps(&B->data[i]);
            __m256 vec_p = ICAN_ARRAY_SIMD_mul_ps(vec_a, vec_b);
            ICAN_ARRAY_SIMD_store_ps(&A->data[i], vec_p);
        } else {
            for (size_t k = i; k < vec_end; k++) {
                A->data[k] = A->data[k] * B->data[k];
            }
        }
    }
}

float array1d_dot_sum(Array1D *A, Array1D *B) {
    ASSERT(A->rows == B->rows);
    float sum = 0.0f;

    #pragma omp parallel
    {
        __m256 vec_sum = ICAN_ARRAY_SIMD_setzero_ps();
        float partial_sum = 0.0f;

        #pragma omp for
        for (size_t i = 0; i < A->rows; i += ICAN_ARRAY_SIMD) {
            size_t vec_end = (i + ICAN_ARRAY_SIMD > A->rows) ? A->rows : i + ICAN_ARRAY_SIMD;
            if (vec_end - i == ICAN_ARRAY_SIMD) {
                __m256 vec_a = ICAN_ARRAY_SIMD_load_ps(&A->data[i]);
                __m256 vec_b = ICAN_ARRAY_SIMD_load_ps(&B->data[i]);
                __m256 vec_p = ICAN_ARRAY_SIMD_mul_ps(vec_a, vec_b);
                vec_sum = ICAN_ARRAY_SIMD_add_ps(vec_sum, vec_p);
            } else {
                for (size_t j = i; j < vec_end; j++) {
                    partial_sum += A->data[j] * B->data[j];
                }
            }
        }

        float vec_result[ICAN_ARRAY_SIMD];
        ICAN_ARRAY_SIMD_store_ps(vec_result, vec_sum);
        for (size_t k = 0; k < ICAN_ARRAY_SIMD; k++) {
            partial_sum += vec_result[k];
        }

        #pragma omp atomic
        sum += partial_sum;
    }

    return sum;
}

float array1d_dot_scalar_sum(Array1D *A, Array1D *B) {
    ASSERT(A->rows == B->rows);
    float sum = 0.0f;

    #pragma omp parallel
    {
        __m256 vec_sum = ICAN_ARRAY_SIMD_setzero_ps();
        float partial_sum = 0.0f;

        #pragma omp for
        for (size_t i = 0; i < A->rows; i += ICAN_ARRAY_SIMD) {
            size_t vec_end = (i + ICAN_ARRAY_SIMD > A->rows) ? A->rows : i + ICAN_ARRAY_SIMD;
            if (vec_end - i == ICAN_ARRAY_SIMD) {
                __m256 vec_a = ICAN_ARRAY_SIMD_load_ps(&A->data[i]);
                __m256 vec_b = ICAN_ARRAY_SIMD_load_ps(&B->data[i]);
                __m256 vec_p = ICAN_ARRAY_SIMD_mul_ps(vec_a, vec_b);
                vec_sum = ICAN_ARRAY_SIMD_add_ps(vec_sum, vec_p);
            } else {
                for (size_t j = i; j < A->rows; j++) {
                    partial_sum += A->data[j] * B->data[j];
                }
            }
        }

        float vec_result[8];
        ICAN_ARRAY_SIMD_store_ps(vec_result, vec_sum);
        for (int k = 0; k < 8; k++) {
            partial_sum += vec_result[k];
        }

        #pragma omp atomic
        sum += partial_sum;
    }

    return sum;
}

Array1D *array1d_mul_scalar(Array1D *A, float scalar) {
    Array1D *result = array1d_alloc(A->rows);

    #pragma omp parallel for
    for (size_t i = 0; i < A->rows; i += ICAN_ARRAY_SIMD) {
        size_t vec_end = (i + ICAN_ARRAY_SIMD > A->rows) ? A->rows : i + ICAN_ARRAY_SIMD;
        if (vec_end - i == ICAN_ARRAY_SIMD) {
            __m256 vec_a = ICAN_ARRAY_SIMD_load_ps(&A->data[i]);
            __m256 vec_scalar = ICAN_ARRAY_SIMD_set1_ps(scalar);
            __m256 vec_res = ICAN_ARRAY_SIMD_mul_ps(vec_a, vec_scalar);
            ICAN_ARRAY_SIMD_store_ps(&result->data[i], vec_res);
        } else {
            for (size_t j = i; j < vec_end; j++) {
                result->data[j] = A->data[j] * scalar;
            }
        }
    }

    return result;
}

void array1d_mul_scalar_inplace(Array1D *A, float scalar) {
    #pragma omp parallel for
    for (size_t i = 0; i < A->rows; i += ICAN_ARRAY_SIMD) {
        size_t vec_end = (i + ICAN_ARRAY_SIMD > A->rows) ? A->rows : i + ICAN_ARRAY_SIMD;
        if (vec_end - i == ICAN_ARRAY_SIMD) {
            __m256 vec_a = ICAN_ARRAY_SIMD_load_ps(&A->data[i]);
            __m256 vec_scalar = ICAN_ARRAY_SIMD_set1_ps(scalar);
            __m256 vec_res = ICAN_ARRAY_SIMD_mul_ps(vec_a, vec_scalar);
            ICAN_ARRAY_SIMD_store_ps(&A->data[i], vec_res);
        } else {
            for (size_t j = i; j < vec_end; j++) {
                A->data[j] = A->data[j] * scalar;
            }
        }
    }
}

Array1D *array1d_fill(Array1D *A, float data) {
    __m256 vec_data = ICAN_ARRAY_SIMD_set1_ps(data);
    Array1D *O = array1d_alloc(A->rows);

    #pragma omp parallel for
    for (size_t i = 0; i < A->rows; i+=ICAN_ARRAY_SIMD) {
        size_t vec_end = (i + ICAN_ARRAY_SIMD > A->rows) ? A->rows : i + ICAN_ARRAY_SIMD;
        if (vec_end - i == ICAN_ARRAY_SIMD) {
            ICAN_ARRAY_SIMD_store_ps(&O->data[i], vec_data);
        }else {
            for (size_t k = i; k < vec_end; k++) {
                O->data[k] = data;
            }
        }
    }
    return O;
}

void array1d_fill_inplace(Array1D *A, float data) {
    __m256 vec_data = ICAN_ARRAY_SIMD_set1_ps(data);

    #pragma omp parallel for
    for (size_t i = 0; i < A->rows; i+=ICAN_ARRAY_SIMD) {
        size_t vec_end = (i + ICAN_ARRAY_SIMD > A->rows) ? A->rows : i + ICAN_ARRAY_SIMD;
        if (vec_end - i == ICAN_ARRAY_SIMD) {
            ICAN_ARRAY_SIMD_store_ps(&A->data[i], vec_data);
        }else {
            for (size_t k = i; k < vec_end; k++) {
                A->data[k] = data;
            }
        }
    }
}

Array1D *array1d_apply(Array1D *A, float (*func)(float)) {
    Array1D *O = array1d_alloc(A->rows);

    #pragma omp parallel for
    for (size_t i = 0; i < A->rows; i+=ICAN_ARRAY_SIMD) {
        size_t vec_end = (i + ICAN_ARRAY_SIMD > A->rows) ? A->rows : i + ICAN_ARRAY_SIMD;
        if (vec_end - i == ICAN_ARRAY_SIMD) {
            __m256 vec_a = ICAN_ARRAY_SIMD_load_ps(&A->data[i]);
            for (size_t k = 0; k < ICAN_ARRAY_SIMD; k++) {
                float val = func(((float*)&vec_a)[k]);
                ((float*)&vec_a)[k] = val;
            }
            ICAN_ARRAY_SIMD_store_ps(&O->data[i], vec_a);
        }else {
            for (size_t k = i; k < vec_end; k++) {
                O->data[k] = func(A->data[k]);
            }
        }
    }
    return O;
}

void array1d_apply_inplace(Array1D *A, float (*func)(float)) {
    #pragma omp parallel for
    for (size_t i = 0; i < A->rows; i+=ICAN_ARRAY_SIMD) {
        size_t vec_end = (i + ICAN_ARRAY_SIMD > A->rows) ? A->rows : i + ICAN_ARRAY_SIMD;
        if (vec_end - i == ICAN_ARRAY_SIMD) {
            __m256 vec_a = ICAN_ARRAY_SIMD_load_ps(&A->data[i]);
            for (size_t k = 0; k < ICAN_ARRAY_SIMD; k++) {
                float val = func(((float*)&vec_a)[k]);
                ((float*)&vec_a)[k] = val;
            }
            ICAN_ARRAY_SIMD_store_ps(&A->data[i], vec_a);
        }else {
            for (size_t k = i; k < vec_end; k++) {
                A->data[k] = func(A->data[k]);
            }
        }
    }
}

Array1D *array1d_sort(Array1D *A, int (*comp)(float, float)) {
    Array1D *O = array1d_alloc(A->rows);
    memcpy(O->data, A->data, A->rows * sizeof(float));

    for (size_t i = 0; i < O->rows - 1; i++) {
        for (size_t j = 0; j < O->rows - i - 1; j++) {
            if (comp(O->data[j], O->data[j + 1]) > 0) {
                float temp = O->data[j];
                O->data[j] = O->data[j + 1];
                O->data[j + 1] = temp;
            }
        }
    }

    return O;
}

void array1d_sort_inplace(Array1D *A, int (*comp)(float, float)) {
    for (size_t i = 0; i < A->rows - 1; i++) {
        for (size_t j = 0; j < A->rows - i - 1; j++) {
            if (comp(A->data[j], A->data[j + 1]) > 0) {
                float temp = A->data[j];
                A->data[j] = A->data[j + 1];
                A->data[j + 1] = temp;
            }
        }
    }
}

float array1d_max(Array1D *A) {
    float max_value = A->data[0];
    #pragma omp parallel for reduction(max:max_value)
    for (size_t i = 0; i < A->rows; i+=ICAN_ARRAY_SIMD) {
        size_t vec_end = (i + ICAN_ARRAY_SIMD > A->rows) ? A->rows : i + ICAN_ARRAY_SIMD;
        if (vec_end - i == ICAN_ARRAY_SIMD) {
            __m256 vec_a = ICAN_ARRAY_SIMD_load_ps(&A->data[i]);
            for (size_t k = 0; k < ICAN_ARRAY_SIMD; k++) {
                max_value = fmaxf(max_value, ((float*)&vec_a)[k]);
            }
        }else {
            for (size_t k = i; k < vec_end; k++) {
                max_value = fmaxf(max_value, A->data[k]);
            }
        }
    }
    return max_value;
}

float array1d_min(Array1D *A) {
    float min_value = A->data[0];
    #pragma omp parallel for reduction(min:min_value)
    for (size_t i = 0; i < A->rows; i+=ICAN_ARRAY_SIMD) {
        size_t vec_end = (i + ICAN_ARRAY_SIMD > A->rows) ? A->rows : i + ICAN_ARRAY_SIMD;
        if (vec_end - i == ICAN_ARRAY_SIMD) {
            __m256 vec_a = ICAN_ARRAY_SIMD_load_ps(&A->data[i]);
            for (size_t k = 0; k < ICAN_ARRAY_SIMD; k++) {
                min_value = fminf(min_value, ((float*)&vec_a)[k]);
            }
        }else {
            for (size_t k = i; k < vec_end; k++) {
                min_value = fminf(min_value, A->data[k]);
            }
        }
    }
    return min_value;
}

Array1D *array1d_slice(Array1D *A, int64 start, int64 end) {
    ASSERT(start >= 0 && end <= A->rows && start < end);
    int64 new_size = end - start;
    Array1D *O = array1d_alloc(new_size);
    memcpy(O->data, &A->data[start], new_size * sizeof(float));
    return O;
}

Array1D *array1d_clone(Array1D *A) {
    Array1D *O = array1d_alloc(A->rows);
    memcpy(O->data, A->data, A->rows * sizeof(float));
    return O;
}

Array1D *array1d_from(float *data, int64 rows) {
    Array1D *O = array1d_alloc(rows);
    memcpy(O->data, data, rows * sizeof(float));
    return O;
}

Array2D *array1d_to_array2d(Array1D *A, int64 rows, int64 cols) {
    Array2D *O = array2d_from(A->data, rows, cols);
    return O;
}

void array1d_print(Array1D *array) {
    printf("Rows = %lld = [", array->rows);
    for (int i = 0; i < array->rows; i++) {
        printf("%f", array->data[i]);
        if(i < array->rows - 1) {
            printf(" ");
        }
    }
    printf("]\n");
}