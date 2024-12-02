#ifndef ICAN_MACHINELEARNING_SUPERVISEDLEARNING_KNN_H

#define ICAN_MACHINELEARNING_SUPERVISEDLEARNING_KNN_H

#define ICAN_USE_ARRAY

#include "ican.h"

struct s_knn {
    Array2D *x;
    Array2D *y;
    int16 k;
};

typedef struct s_knn KNN;

KNN *knn_alloc(Array2D *x, Array2D *y, int16 k);
void knn_free(KNN *knn);
float knn_predict(KNN *knn, Array1D *x);

#endif // !ICAN_MACHINELEARNING_SUPERVISEDLEARNING_KNN_H