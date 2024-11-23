#include "src/ican.h"

int main() {
  size_t embed_dim = 512;
  size_t num_heads = 8;
  size_t seq_len = 10;
  bool is_mask = true;

  Encoder *mha = transformers_encoder_alloc(embed_dim, num_heads, seq_len, is_mask);


    Iray2D *input = iray2d_alloc(seq_len, embed_dim / num_heads);
  Iray1D *output = transformers_encoder_forward(mha, input);

  printf("Output (first 5 values):\n");
  for (size_t i = 0; i < 5; i++) {
    printf("%f ", output->data[i]);
  }
  printf("\n");

  transformers_encoder_free(mha);
  iray1d_free(output);
  iray2d_free(input);

  return 0;
}