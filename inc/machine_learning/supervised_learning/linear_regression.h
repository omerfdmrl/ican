#ifndef ICAN_MACHINELEARNING_SUPERVISEDLEARNING_LINEARREGRESSION_H

#define ICAN_MACHINELEARNING_SUPERVISEDLEARNING_LINEARREGRESSION_H

#define ICAN_USE_ARRAY

#include "ican.h"

struct s_linear_regression {
    Array1D *w;
    float b;
    float lr;
};

typedef struct s_linear_regression LinearRegression;

LinearRegression *linear_regression_alloc();
void linear_regression_free(LinearRegression *lr);
float linear_regression_predict(LinearRegression *lr, Array1D *x);
void linear_regression_fit(LinearRegression *lr, Array2D *x, Array2D *y, int32 epochs);

#endif // !ICAN_MACHINELEARNING_SUPERVISEDLEARNING_LINEARREGRESSION_H