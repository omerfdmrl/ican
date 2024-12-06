#include "machine_learning/supervised_learning/logistic_regression.h"

LogisticRegression *logistic_regression_alloc() {
    LogisticRegression *lr = (LogisticRegression *)ICAN_MALLOC(sizeof(LogisticRegression));
    lr->w = NULL;
    lr->b = 0;
    lr->lr = 1e-3;
    return lr;
}

void logistic_regression_free(LogisticRegression *lr) {
    array1d_free(lr->w);
    ICAN_FREE(lr);
}

float logistic_regression_feed_forward(LogisticRegression *lr, Array1D *x) {
    float z = array1d_dot_scalar_sum(x, lr->w) + lr->b;
    return 1 / (1 + expf(-z));
}

float logistic_regression_cost(LogisticRegression *lr, Array2D *x, Array2D *y) {
    float cost = 0.0;
    for (size_t i = 0; i < x->rows; i++) {
        Array1D *xi = array1d_from(&x->data[i * x->cols], x->cols);
        float predicted = logistic_regression_predict(lr, xi);
        array1d_free(xi);
        cost += powf(y->data[i] - predicted, 2);
    }
    return cost / x->rows;
}

float logistic_regression_predict(LogisticRegression *lr, Array1D *x) {
    return logistic_regression_feed_forward(lr, x) >= 0.5 ? 1 : 0;
}

void logistic_regression_fit(LogisticRegression *lr, Array2D *x, Array2D *y, int32 epochs) {
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
            float y_pred = logistic_regression_feed_forward(lr, xi);
            float error = y->data[i] - y_pred;
            float dz = y_pred * (1 - y_pred) * error;

            Array1D *scalar = array1d_mul_scalar(xi, dz);
            array1d_add_inplace(dw, scalar);
            db += dz;

            array1d_free(xi);
            array1d_free(scalar);
        }
        array1d_mul_scalar_inplace(dw, -2 * lr->lr / x->rows);
        array1d_sub_inplace(lr->w, dw);
        lr->b -= -2 * lr->lr * db / x->rows;
    }
    array1d_free(dw);
}
