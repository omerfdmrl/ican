#include "machine_learning/supervised_learning/svm.h"

SVM *svm_alloc() {
    SVM *svm = (SVM *)ICAN_MALLOC(sizeof(SVM));
    svm->lr = 1e-3;
    svm->c = 1e-1;
    svm->b = 0;
    svm->w = NULL;
    array1d_fill_inplace(svm->w, 0);
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

void svm_fit(SVM *svm, Array2D *x, Array2D *y, int32 epochs) {
    if (svm->w == NULL) {
        svm->w = array1d_alloc(x->cols);
    }
    for (size_t i = 0; i < epochs; i++) {
        for (size_t j = 0; j < x->rows; j++) {
            Array1D *xi = array1d_from(&x->data[j], x->cols);
            float yi = y->data[j];
            bool condition = y->data[j] * _predict(svm, xi) >= 1;
            if (condition) {
                Array1D *dw = array1d_mul_scalar(svm->w, 2 * svm->c);
                array1d_mul_scalar(dw, svm->lr);
                array1d_sub_inplace(svm->w, dw);
            }else {
                Array1D *dw = array1d_mul_scalar(svm->w, 2 * svm->c);
                Array1D *yx = array1d_mul_scalar(xi, yi);
                array1d_sub_inplace(dw, yx);
                array1d_mul_scalar_inplace(dw, svm->lr);
                array1d_sub_inplace(svm->w, dw);
                array1d_free(dw);
                array1d_free(yx);
                svm->b -= svm->lr * yi;
            }
            array1d_free(xi);
        }
    }
}
