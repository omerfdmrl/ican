/**
 * make run SOURCE=./examples/embedding.c
 */

#include "../src/ican.h"
#include <time.h>
int main() {
    srand(50);
    const size_t embed_dim = 16;
    const size_t vocab_size = 10;


    TransformerEmbedding *e = transform_embedding_alloc(vocab_size, embed_dim);
    iray2d_print(e->W);
    float *o = transform_embedding_forward(e, 3);
    for (size_t i = 0; i < embed_dim; i++){
        printf("O[%zu] = %f\n", i, o[i]);
    }
    transform_embedding_free(e);
}