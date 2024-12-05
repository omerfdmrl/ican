#ifndef ICAN_H

#define ICAN_H

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>

typedef unsigned char int8; // 0-255
typedef unsigned short int int16;  // 0-65535
typedef unsigned int int32; // 0-4294967295
typedef unsigned long long int int64; // 0-18446744073709551615

#ifndef ICAN_MALLOC
#define ICAN_MALLOC malloc
#endif // !ICAN_MALLOC

#ifndef ICAN_FREE
#define ICAN_FREE free
#endif // !ICAN_FREE

#ifndef ICAN_CALLOC
#define ICAN_CALLOC calloc
#endif // !ICAN_CALLOC

#ifndef ICAN_REALLOC
#define ICAN_REALLOC realloc
#endif // !ICAN_REALLOC

#include "logger.h"

#ifdef ICAN_USE_ARRAY
#include "data_structure/array.h"
#endif // ICAN_USE_ARRAY

#ifdef ICAN_USE_LINKED_LIST
#include "data_structure/linked_list.h"
#endif // !ICAN_USE_LINKED_LIST

#ifdef ICAN_USE_KNN
#include "machine_learning/supervised_learning/knn.h"
#endif // !ICAN_USE_KNN

#ifdef ICAN_USE_SVM
#include "machine_learning/supervised_learning/svm.h"
#endif // !ICAN_USE_SVM

#ifdef ICAN_USE_LINEAR_REGRESSION
#include "machine_learning/supervised_learning/linear_regression.h"
#endif // !ICAN_USE_LINEAR_REGRESSION

#endif // !ICAN_H