#include "machine_learning/supervised_learning/svm.h"

SVM *svm_alloc(Array2D *x, Array2D *y) {
    SVM *svm = (SVM *)malloc(sizeof(SVM));
    svm->x = x;
    svm->y = y;
    svm->lr = 1e-3;
    svm->c = 1e-1;
    svm->b = 0;
    svm->w = array1d_alloc(x->cols);
    array1d_fill(svm->w, 0);
    return svm;
}

void svm_free(SVM *svm) {
    array1d_free(svm->w);
    ICAN_FREE(svm);
}

float _predict(SVM *svm, Array1D *x) {
    float dot_product = array1d_dot_scalar(x, svm->w);
    return dot_product + svm->b;
}

int svm_predict(SVM *svm, Array1D *x) {
    float predicted = _predict(svm, x);
    return predicted > 0 ? 1 : -1;
}

void svm_fit(SVM *svm, int32 epochs) {
    for (size_t i = 0; i < epochs; i++) {
        for (size_t j = 0; j < svm->x->rows; j++) {
            Array1D *x = array1d_from(&svm->x->data[j], svm->x->cols);
            float y = svm->y->data[j];
            bool condition = svm->y->data[j] * _predict(svm, x) >= 1;
            if (condition) {
                for (size_t k = 0; k < svm->w->rows; k++) {
                    svm->w->data[k] -= svm->lr * (2 * svm->c * svm->w->data[k]);
                }
            }else {
                for (size_t k = 0; k < svm->w->rows; k++) {
                    svm->w->data[k] -= svm->lr * (2 * svm->c * svm->w->data[k] - y * x->data[k]);
                }
                svm->b -= svm->lr * y;
            }
            array1d_free(x);
        }
    }
    
}
