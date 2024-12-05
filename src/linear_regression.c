#include "machine_learning/supervised_learning/linear_regression.h"

LinearRegression *linear_regression_alloc() {
    LinearRegression *lr = (LinearRegression *)ICAN_MALLOC(sizeof(LinearRegression));
    lr->w = NULL;
    lr->b = 0;
    lr->lr = 1e-3;
    return lr;
}

void linear_regression_free(LinearRegression *lr) {
    array1d_free(lr->w);
    ICAN_FREE(lr);
}

float linear_regression_predict(LinearRegression *lr, Array1D *x) {
    float dotted = array1d_dot_scalar_sum(x, lr->w);
    return dotted + lr->b;
}

void linear_regression_fit(LinearRegression *lr, Array2D *x, Array2D *y, int32 epochs) {
    if (lr->w == NULL) {
        lr->w = array1d_alloc(x->cols);
        array1d_fill_inplace(lr->w, 0);
    }
    Array1D *dw = array1d_alloc(lr->w->rows);
    for (size_t epoch = 0; epoch < epochs; epoch++) {
        array1d_fill_inplace(dw, 0);
        float db = 0;
        for (size_t i = 0; i < x->rows; i++) {
            Array1D *xi = array1d_from(&x->data[i * x->cols], x->cols);
            float y_pred = linear_regression_predict(lr, xi);
            float error = y->data[i] - y_pred;

            Array1D *scalar = array1d_mul_scalar(xi, error);
            array1d_add_inplace(dw, scalar);
            db += error;

            

            array1d_free(xi);
            array1d_free(scalar);
        }
        array1d_mul_scalar_inplace(dw, -2 * lr->lr / x->rows);
        array1d_sub_inplace(lr->w, dw);
        lr->b -= -2 * lr->lr * db / x->rows;
    }
    array1d_free(dw);
}
