#ifndef ICAN_MACHINELEARNING_SUPERVISEDLEARNING_SVM_H

#define ICAN_MACHINELEARNING_SUPERVISEDLEARNING_SVM_H

#define ICAN_USE_ARRAY

#include "ican.h"

struct s_svm {
    float lr;
    float c;
    Array1D *w;
    float b;
};

typedef struct s_svm SVM;

SVM *svm_alloc();
void svm_free(SVM *svm);
int svm_predict(SVM *svm, Array1D *x);
void svm_fit(SVM *svm, Array2D *x, Array2D *y, int32 epochs);

#endif // !ICAN_MACHINELEARNING_SUPERVISEDLEARNING_SVM_H