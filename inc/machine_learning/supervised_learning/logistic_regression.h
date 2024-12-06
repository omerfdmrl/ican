#ifndef ICAN_MACHINELEARNING_SUPERVISEDLEARNING_LOGISTICREGRESSION_H

#define ICAN_MACHINELEARNING_SUPERVISEDLEARNING_LOGISTICREGRESSION_H

#define ICAN_USE_ARRAY

#include "ican.h"

struct s_logistic_regression {
    Array1D *w;
    float b;
    float lr;
};

typedef struct s_logistic_regression LogisticRegression;

LogisticRegression *logistic_regression_alloc();
void logistic_regression_free(LogisticRegression *lr);
float logistic_regression_cost(LogisticRegression *lr, Array2D *x, Array2D *y);
float logistic_regression_predict(LogisticRegression *lr, Array1D *x);
void logistic_regression_fit(LogisticRegression *lr, Array2D *x, Array2D *y, int32 epochs);

#endif // !ICAN_MACHINELEARNING_SUPERVISEDLEARNING_LOGISTICREGRESSION_H