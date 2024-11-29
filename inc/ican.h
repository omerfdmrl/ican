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

#include "iray.h"

#include "logger.h"

#endif // !ICAN_H