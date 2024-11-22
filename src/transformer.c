#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "ican.h"

Iray2D *create_causal_mask(size_t seq_len) {
  Iray2D *mask = iray2d_alloc(seq_len, seq_len);
  for (size_t i = 0; i < seq_len; i++) {
    for (size_t j = 0; j < seq_len; j++) {
        if (j > i) {
            mask->data[i][j] = -INFINITY;
        } else {
            mask->data[i][j] = 0.0f;
        }
    }
  }
  return mask;
}

ScaledDotProductAttention *transformers_sdpa_alloc(size_t seq_len, size_t head_dim, size_t embed_dim, bool is_mask) {
  ScaledDotProductAttention *sdpa = (ScaledDotProductAttention *)malloc(sizeof(ScaledDotProductAttention));
  sdpa->query = iray2d_alloc(seq_len, head_dim);
  sdpa->key = iray2d_alloc(seq_len, head_dim);
  sdpa->value = iray2d_alloc(seq_len, head_dim);
  sdpa->embed_dim = embed_dim;
  sdpa->head_dim = head_dim;
  sdpa->seq_len = seq_len;
  sdpa->is_mask = is_mask;
  sdpa->W_q = iray2d_alloc(embed_dim, head_dim);
  sdpa->W_k = iray2d_alloc(embed_dim, head_dim);
  sdpa->W_v = iray2d_alloc(embed_dim, head_dim);
  return sdpa;
}
void transformers_sdpa_free(ScaledDotProductAttention *sdpa) {
  iray2d_free(sdpa->query);
  iray2d_free(sdpa->key);
  iray2d_free(sdpa->value);
  free(sdpa);
}
Iray2D *transformers_sdpa_forward(ScaledDotProductAttention *sdpa) {
  Iray2D *key_transposed = iray2d_transpose(sdpa->key);
  Iray2D *matmul = iray2d_dot(sdpa->query, key_transposed);
  iray2d_free(key_transposed);
  Iray2D *scaled = iray2d_scale(matmul, sqrtf((float)sdpa->embed_dim));
  iray2d_free(matmul);

  if(sdpa->is_mask) {
    Iray2D *mask = create_causal_mask(sdpa->seq_len);
    Iray2D *temp = scaled;
    scaled = iray2d_add(scaled, mask);
    iray2d_free(temp);
  }

  Iray2D *softmaxed = iray2d_softmax(scaled, 1);
  iray2d_free(scaled);

  Iray2D *output = iray2d_dot(softmaxed, sdpa->value);
  iray2d_free(softmaxed);

  return output;
}

MultiHeadAttention *transformers_mha_alloc(size_t embed_dim, size_t num_heads, size_t seq_len, bool is_mask) {
  MultiHeadAttention *mha = (MultiHeadAttention *)malloc(sizeof(MultiHeadAttention));
  mha->sdpa = (ScaledDotProductAttention **)malloc(sizeof(ScaledDotProductAttention *) * num_heads);
  mha->embed_dim = embed_dim;
  mha->num_heads = num_heads;
  mha->seq_len = seq_len;
  mha->W_o = iray2d_alloc(embed_dim, embed_dim);

  for (size_t i = 0; i < num_heads; i++) {
    mha->sdpa[i] = transformers_sdpa_alloc(seq_len, embed_dim / num_heads, embed_dim, is_mask);
  }
  return mha;
}
void transformers_mha_free(MultiHeadAttention *mha) {
  for (size_t i = 0; i < mha->num_heads; i++) {
    transformers_sdpa_free(mha->sdpa[i]);
  }
  free(mha);
}
Iray2D *transformers_mha_forward(MultiHeadAttention *mha) {
  Iray2D **sdpa_outputs = (Iray2D **)malloc(sizeof(Iray2D *) * mha->num_heads);

  for (size_t i = 0; i < mha->num_heads; i++) {
    mha->sdpa[i]->query = iray2d_dot(mha->sdpa[i]->query, iray2d_transpose(mha->sdpa[i]->W_q));
    mha->sdpa[i]->key = iray2d_dot(mha->sdpa[i]->key, iray2d_transpose(mha->sdpa[i]->W_k));
    mha->sdpa[i]->value = iray2d_dot(mha->sdpa[i]->value, iray2d_transpose(mha->sdpa[i]->W_v));
    sdpa_outputs[i] = transformers_sdpa_forward(mha->sdpa[i]);
  }
  Iray2D *concated = iray2d_concat(sdpa_outputs, mha->num_heads);
  printf("ROW = %zu, COL = %zu\n", concated->rows, concated->cols);
  printf("OUTING\n");
  Iray2D *output = iray2d_dot(concated, iray2d_transpose(mha->W_o));
  printf("OUTED\n");

  iray2d_free(concated);
  for (size_t i = 0; i < mha->num_heads; i++) {
    iray2d_free(sdpa_outputs[i]);
  }
  free(sdpa_outputs);

  return output;
}

Encoder *transformers_encoder_alloc() {}

#endif // !TRANSFORMER_H