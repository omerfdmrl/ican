#include "iray.h"

Iray2D *iray2d_alloc(int64 rows, int64 cols) {
    Iray2D *iray = (Iray2D *) ICAN_MALLOC(sizeof(Iray2D));
    ASSERT(iray != NULL);
    iray->rows = rows;
    iray->cols = cols;
    iray->data = (float *) aligned_alloc(32, sizeof(float) * rows * cols);
    ASSERT(iray->data != NULL);
    return iray;
}

void iray2d_free(Iray2D *iray) {
    if(iray != NULL) {
        if (iray->data != NULL) {
            ICAN_FREE(iray->data);
        }
        ICAN_FREE(iray);
    }
}

Iray2D *iray2d_add(Iray2D *A, Iray2D *B) {
    ASSERT(A->cols == B->cols);
    ASSERT(A->rows == B->rows);
    Iray2D *O = iray2d_alloc(A->rows, A->cols);
    
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < A->rows; i++) {
        for (size_t j = 0; j < A->cols; j += 8) {
            size_t vec_end = (j + 8 > A->cols) ? A->cols : j + 8;
            if (vec_end - j == 8) {
                __m256 vec_a = _mm256_load_ps(&A->data[i * A->cols + j]);
                __m256 vec_b = _mm256_load_ps(&B->data[i * B->cols + j]);
                __m256 vec_sum = _mm256_add_ps(vec_a, vec_b);
                _mm256_store_ps(&O->data[i * O->cols + j], vec_sum);
            } else {
                for (size_t k = j; k < vec_end; k++) {
                    O->data[i * O->cols + k] = A->data[i * A->cols + k] + B->data[i * B->cols + k];
                }
            }
        }
    }
    return O;
}

Iray2D *iray2d_dot(Iray2D *A, Iray2D *B) {
    ASSERT(A->cols == B->rows);
    Iray2D *O = iray2d_alloc(A->rows, B->cols);

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < A->rows; i++) {
        for (size_t j = 0; j < B->cols; j++) {
            O->data[i * O->cols + j] = 0.0f;
            for (size_t k = 0; k < A->cols; k+=8) {
                size_t vec_end = (k + 8 > A->cols) ? A->cols : k + 8;
                if (vec_end - k == 8) {
                    __m256 vec_a = _mm256_load_ps(&A->data[i * A->cols + k]);
                    __m256 vec_b = _mm256_load_ps(&B->data[k * B->cols + j]);
                    __m256 vec_p = _mm256_mul_ps(vec_a, vec_b);
                    __m256 vec_s = _mm256_hadd_ps(vec_p, vec_p);
                    vec_s = _mm256_hadd_ps(vec_s, vec_s);
                    float result[8];
                    _mm256_store_ps(result, vec_s);
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

Iray2D *iray2d_apply(Iray2D *A, float (*func)(float)) {
    Iray2D *O = iray2d_alloc(A->rows, A->cols);
    
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < A->rows; i++) {
        for (size_t j = 0; j < A->cols; j += 8) {
            size_t vec_end = (j + 8 > A->cols) ? A->cols : j + 8;
            if (vec_end - j == 8) {
                __m256 vec_a = _mm256_load_ps(&A->data[i * A->cols + j]);
                for (size_t k = 0; k < 8; k++) {
                    float val = func(((float*)&vec_a)[k]);
                    ((float*)&vec_a)[k] = val;
                }
                _mm256_store_ps(&O->data[i * O->cols + j], vec_a);
            } else {
                for (size_t k = j; k < vec_end; k++) {
                    O->data[i * O->cols + k] = func(A->data[i * A->cols + k]);
                }
            }
        }
    }
    return O;
}

Iray2D *iray2d_scale(Iray2D *A, float scalar) {
    __m256 vec_scalar = _mm256_set1_ps(scalar);
    Iray2D *O = iray2d_alloc(A->rows, A->cols);
    
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < A->rows; i++) {
        for (size_t j = 0; j < A->cols; j += 8) {
            size_t vec_end = (j + 8 > A->cols) ? A->cols : j + 8;
            if (vec_end - j == 8) {
                __m256 vec_a = _mm256_load_ps(&A->data[i * A->cols + j]);
                __m256 vec_o = _mm256_mul_ps(vec_a, vec_scalar);
                _mm256_store_ps(&O->data[i * O->cols + j], vec_o);
            } else {
                for (size_t k = j; k < vec_end; k++) {
                    O->data[i * O->cols + k] = A->data[i * A->cols + k] * scalar;
                }
            }
        }
    }
    return O;
}

void iray2d_fill(Iray2D *A, float data) {
    __m256 vec_data = _mm256_set1_ps(data);
    
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < A->rows; i++) {
        for (size_t j = 0; j < A->cols; j += 8) {
            size_t vec_end = (j + 8 > A->cols) ? A->cols : j + 8;
            if (vec_end - j == 8) {
                _mm256_store_ps(&A->data[i * A->cols + j], vec_data);
            } else {
                for (size_t k = j; k < vec_end; k++) {
                    A->data[i * A->cols + k] = data;
                }
            }
        }
    }
}

float iray2d_max(Iray2D *A) {
    float max_value = A->data[0];
    #pragma omp parallel for reduction(max:max_value) collapse(2)
    for (size_t i = 0; i < A->rows; i++) {
        for (size_t j = 0; j < A->cols; j += 8) {
            size_t vec_end = (j + 8 > A->cols) ? A->cols : j + 8;
            if (vec_end - j == 8) {
                __m256 vec_a = _mm256_load_ps(&A->data[i * A->cols + j]);
                for (size_t k = 0; k < 8; k++) {
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

float iray2d_min(Iray2D *A) {
    float min_value = A->data[0];
    #pragma omp parallel for reduction(min:min_value) collapse(2)
    for (size_t i = 0; i < A->rows; i++) {
        for (size_t j = 0; j < A->cols; j += 8) {
            size_t vec_end = (j + 8 > A->cols) ? A->cols : j + 8;
            if (vec_end - j == 8) {
                __m256 vec_a = _mm256_load_ps(&A->data[i * A->cols + j]);
                for (size_t k = 0; k < 8; k++) {
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

Iray2D *iray2d_clone(Iray2D *A) {
    Iray2D *O = iray2d_alloc(A->rows, A->cols);
    memcpy(O->data, A->data, A->rows * A->cols * sizeof(float));
    return O;
}

Iray2D *iray2d_transpose(Iray2D *A) {
    Iray2D *O = iray2d_alloc(A->cols, A->rows);
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < A->rows; i++) {
        for (size_t j = 0; j < A->cols; j++) {
            size_t vec_end = (j + 8 > A->cols) ? A->cols : j + 8;
            if (vec_end - j == 8) {
                __m256 vec_a = _mm256_load_ps(&A->data[i * A->cols + j]);
                for (size_t k = 0; k < 8; k++) {
                    O->data[(j + k) * O->cols + i] = ((float *)&vec_a)[k];
                }
            }else {
                for (size_t k = j; k < vec_end; k++) {
                    O->data[k * O->cols + i] = A->data[i * A->cols + k];
                }
            }
        }
    }
    return O;
}

void iray2d_print(Iray2D *iray) {
    for (int i = 0; i < iray->rows; i++) {
        for (int j = 0; j < iray->cols; j++) {
            printf("%f ", iray->data[i * iray->cols + j]);
        }
        printf("\n");
    }
}

Iray1D *iray1d_alloc(int64 rows) {
    Iray1D *iray = (Iray1D *) ICAN_MALLOC(sizeof(Iray1D));
    ASSERT(iray != NULL);
    iray->rows = rows;
    iray->data = (float *) aligned_alloc(32, sizeof(float) * rows);
    ASSERT(iray->data != NULL);
    return iray;
}

void iray1d_free(Iray1D *iray) {
    if(iray != NULL) {
        if (iray->data != NULL) {
            ICAN_FREE(iray->data);
        }
        ICAN_FREE(iray);
    }
}

Iray1D *iray1d_add(Iray1D *A, Iray1D *B) {
    ASSERT(A->rows == B->rows);
    Iray1D *O = iray1d_alloc(A->rows);

    #pragma omp parallel for
    for (size_t i = 0; i < A->rows; i+=8) {
        size_t vec_end = (i + 8 > A->rows) ? A->rows : i + 8;
        if (vec_end - i == 8) {
            __m256 vec_a = _mm256_load_ps(&A->data[i]);
            __m256 vec_b = _mm256_load_ps(&B->data[i]);
            __m256 vec_sum = _mm256_add_ps(vec_a, vec_b);
            _mm256_store_ps(&O->data[i], vec_sum);
        } else {
            for (size_t k = i; k < vec_end; k++) {
                O->data[k] = A->data[k] + B->data[k];
            }
        }
    }
    return O;
}

Iray1D *iray1d_dot(Iray1D *A, Iray1D *B) {
    ASSERT(A->rows == B->rows);
    Iray1D *O = iray1d_alloc(A->rows);

    #pragma omp parallel for
    for (size_t i = 0; i < A->rows; i+=8) {
        size_t vec_end = (i + 8 > A->rows) ? A->rows : i + 8;
        if (vec_end - i == 8) {
            __m256 vec_a = _mm256_load_ps(&A->data[i]);
            __m256 vec_b = _mm256_load_ps(&B->data[i]);
            __m256 vec_p = _mm256_mul_ps(vec_a, vec_b);
            _mm256_store_ps(&O->data[i], vec_p);
        } else {
            for (size_t k = i; k < vec_end; k++) {
                O->data[k] = A->data[k] * B->data[k];
            }
        }
    }
    return O;
}

void iray1d_fill(Iray1D *A, float data) {
    __m256 vec_data = _mm256_set1_ps(data);

    #pragma omp parallel for
    for (size_t i = 0; i < A->rows; i+=8) {
        size_t vec_end = (i + 8 > A->rows) ? A->rows : i + 8;
        if (vec_end - i == 8) {
            _mm256_store_ps(&A->data[i], vec_data);
        }else {
            for (size_t k = i; k < vec_end; k++) {
                A->data[k] = data;
            }
        }
    }
}

Iray1D *iray1d_apply(Iray1D *A, float (*func)(float)) {
    Iray1D *O = iray1d_alloc(A->rows);

    #pragma omp parallel for
    for (size_t i = 0; i < A->rows; i+=8) {
        size_t vec_end = (i + 8 > A->rows) ? A->rows : i + 8;
        if (vec_end - i == 8) {
            __m256 vec_a = _mm256_load_ps(&A->data[i]);
            for (size_t k = 0; k < 8; k++) {
                float val = func(((float*)&vec_a)[k]);
                ((float*)&vec_a)[k] = val;
            }
            _mm256_store_ps(&O->data[i], vec_a);
        }else {
            for (size_t k = i; k < vec_end; k++) {
                O->data[k] = func(A->data[k]);
            }
        }
    }
    return O;
}

Iray1D *iray1d_scale(Iray1D *A, float scalar) {
    __m256 vec_scalar = _mm256_set1_ps(scalar);
    Iray1D *O = iray1d_alloc(A->rows);

    #pragma omp parallel for
    for (size_t i = 0; i < A->rows; i+=8) {
        size_t vec_end = (i + 8 > A->rows) ? A->rows : i + 8;
        if (vec_end - i == 8) {
            __m256 vec_a = _mm256_load_ps(&A->data[i]);
            __m256 vec_o = _mm256_mul_ps(vec_a, vec_scalar);
            _mm256_store_ps(&O->data[i], vec_o);
        }else {
            for (size_t k = i; k < vec_end; k++) {
                A->data[k] = A->data[k] * scalar;
            }
        }
    }
    return O;
}

float iray1d_max(Iray1D *A) {
    float max_value = A->data[0];
    #pragma omp parallel for reduction(max:max_value)
    for (size_t i = 0; i < A->rows; i+=8) {
        size_t vec_end = (i + 8 > A->rows) ? A->rows : i + 8;
        if (vec_end - i == 8) {
            __m256 vec_a = _mm256_load_ps(&A->data[i]);
            for (size_t k = 0; k < 8; k++) {
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

float iray1d_min(Iray1D *A) {
    float min_value = A->data[0];
    #pragma omp parallel for reduction(min:min_value)
    for (size_t i = 0; i < A->rows; i+=8) {
        size_t vec_end = (i + 8 > A->rows) ? A->rows : i + 8;
        if (vec_end - i == 8) {
            __m256 vec_a = _mm256_load_ps(&A->data[i]);
            for (size_t k = 0; k < 8; k++) {
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

Iray1D *iray1d_slice(Iray1D *A, int64 start, int64 end) {
    int64 new_size = end - start;
    size_t j = 0;
    Iray1D *O = iray1d_alloc(new_size);
    #pragma omp parallel for
    for (size_t i = start; i < end; i+=8) {
        size_t vec_end = (i + 8 > A->rows) ? A->rows : i + 8;
        if (vec_end - i == 8) {
            __m256 vec_a = _mm256_load_ps(&A->data[i]);
            _mm256_store_ps(&O->data[j], vec_a);
        }else {
            for (size_t k = i; k < vec_end; k++, j++) {
                O->data[j] = A->data[i];
            }
        }
        j+=8;
    }
    return O;
}

Iray1D *iray1d_clone(Iray1D *A) {
    Iray1D *O = iray1d_alloc(A->rows);
    memcpy(O->data, A->data, A->rows * sizeof(float));
    return O;
}

void iray1d_print(Iray1D *iray) {
    for (int i = 0; i < iray->rows; i++) {
        printf("%f ", iray->data[i]);
    }
    printf("\n");
}