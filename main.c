#define ICAN_USE_IRAY
#include "ican/ican.h"

int main() {
    Iray1D *a = iray1d_alloc(5);
    Iray1D *b = iray1d_alloc(5);
    iray1d_fill(a, 3);
    iray1d_fill(b, 5);
    Iray1D *o = iray1d_dot(a, b);

    iray1d_print(o);

    iray1d_free(a);
    iray1d_free(b);
    iray1d_free(o);
}