/**
 * make run SOURCE=./examples/transformer.c
 */

#include "./src/ican.h"
#include <time.h>
int main() {
    srand(time(0));
    const size_t seq_len = 5;
    const size_t embed_dim = 16;
    const size_t num_heads = 8;

    Iray2D *input = iray2d_alloc(seq_len, embed_dim);
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < embed_dim; j++) {
            input->data[i][j] = (float)rand() / RAND_MAX;
        }
    }

    printf("Input Matrix:\n");
    iray2d_print(input);

    Encoder *encoder = transformer_encoder_alloc(embed_dim,num_heads, seq_len, true);

    Iray2D *output = transformer_encoder_forward(encoder, input);

    printf("\nOutput Matrix:\n");
    iray2d_print(output);

    iray2d_free(input);
    iray2d_free(output);
    // transformer_encoder_free(encoder);

    return 0;
}