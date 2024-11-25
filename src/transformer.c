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
  iray2d_free(sdpa->W_q);
  iray2d_free(sdpa->W_k);
  iray2d_free(sdpa->W_v);
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
  mha->W_o = iray2d_alloc(embed_dim * num_heads, embed_dim);

  for (size_t i = 0; i < num_heads; i++) {
    mha->sdpa[i] = transformers_sdpa_alloc(seq_len, embed_dim / num_heads, embed_dim, is_mask);
  }
  return mha;
}
void transformers_mha_free(MultiHeadAttention *mha) {
  for (size_t i = 0; i < mha->num_heads; i++) {
    transformers_sdpa_free(mha->sdpa[i]);
  }
  iray2d_free(mha->W_o);
  free(mha->sdpa);
  free(mha);
}
Iray2D *transformers_mha_forward(MultiHeadAttention *mha) {
  Iray2D **sdpa_outputs = (Iray2D **)malloc(sizeof(Iray2D *) * mha->num_heads);
  for (size_t i = 0; i < mha->num_heads; i++) {
    mha->sdpa[i]->query = iray2d_dot(mha->sdpa[i]->query, mha->sdpa[i]->W_q);
    mha->sdpa[i]->key = iray2d_dot(mha->sdpa[i]->key, mha->sdpa[i]->W_k);
    mha->sdpa[i]->value = iray2d_dot(mha->sdpa[i]->value, mha->sdpa[i]->W_v);
    sdpa_outputs[i] = transformers_sdpa_forward(mha->sdpa[i]);
  }
  Iray2D *concated = iray2d_concat(sdpa_outputs, mha->num_heads);
  Iray2D *output = iray2d_dot(concated, iray2d_transpose(mha->W_o));
  iray2d_free(concated);
  for (size_t i = 0; i < mha->num_heads; i++) {
    iray2d_free(sdpa_outputs[i]);
  }
  free(sdpa_outputs);

  return output;
}

Norm *transformers_norm_alloc(size_t embed_dim) {
  Norm *norm = malloc(sizeof(Norm));
  norm->embed_dim = embed_dim;
  norm->gamma = malloc(embed_dim * sizeof(float));
  norm->beta = malloc(embed_dim * sizeof(float));

  for (size_t i = 0; i < embed_dim; i++) {
      norm->gamma[i] = 1.0f;
      norm->beta[i] = 0.0f;
  }

  return norm;
}
void transformers_norm_free(Norm *norm) {
  free(norm->gamma);
  free(norm->beta);
  free(norm);
}
Iray1D *transformers_norm_forward(Norm *ln, Iray1D *input, size_t seq_len, size_t embed_dim) {
  Iray1D *output = iray1d_alloc(seq_len * embed_dim);
  for (size_t i = 0; i < seq_len; i++) {
    float mean = 0.0f;
    for (size_t j = 0; j < embed_dim; j++) {
        mean += input->data[i * embed_dim + j];
    }
    mean /= embed_dim;

    float variance = 0.0f;
    for (size_t j = 0; j < embed_dim; j++) {
        variance += (input->data[i * embed_dim + j] - mean) * (input->data[i * embed_dim + j] - mean);
    }
    variance /= embed_dim;

    for (size_t j = 0; j < embed_dim; j++) {
        output->data[i * embed_dim + j] = (input->data[i * embed_dim + j] - mean) / sqrtf(variance + 1e-3f);
        output->data[i * embed_dim + j] = output->data[i * embed_dim + j] * ln->gamma[j] + ln->beta[j];
    }
  }
  return output;
}

Encoder *transformers_encoder_alloc(size_t embed_dim, size_t num_heads, size_t seq_len, bool is_mask) {
  Encoder *encoder = (Encoder *)malloc(sizeof(Encoder));
  encoder->mha = transformers_mha_alloc(embed_dim, num_heads, seq_len, is_mask);
  encoder->norm = transformers_norm_alloc(embed_dim);
  Layer *ld1 = layer_dense(embed_dim, 4 * embed_dim);
  Layer *la1 = layer_activation(RELU);
  Layer *ld2 = layer_dense(4 * embed_dim, embed_dim);
  Model *model = model_alloc(3);
  model_add(model, ld1);
  model_add(model, la1);
  model_add(model, ld2);
  encoder->feed_forward = model;
  encoder->embed_dim = embed_dim;
  encoder->num_heads = num_heads;
  encoder->seq_len = seq_len;
  model_randomize(RandomNormal, encoder->feed_forward);
  return encoder;
}
Iray1D *transformers_encoder_forward(Encoder *encoder, Iray2D *input) {
  ASSERT_MSG(input->rows == encoder->seq_len && input->cols == encoder->embed_dim, "Input dimensions do not match encoder configuration");

  for (size_t i = 0; i < encoder->num_heads; i++) {
      encoder->mha->sdpa[i]->query = input;
      encoder->mha->sdpa[i]->key = input;
      encoder->mha->sdpa[i]->value = input;
  }
  Iray2D *output = transformers_mha_forward(encoder->mha);

  Iray1D *flatted = iray2d_flatten(output);
  Iray1D *norm_output = transformers_norm_forward(encoder->norm, flatted, encoder->seq_len, encoder->embed_dim);
  
  for (size_t i = 0; i < encoder->seq_len; i++) {
      for (size_t j = 0; j < encoder->embed_dim; j++) {
          norm_output->data[i * encoder->embed_dim + j] += input->data[i][j];
      }
  }

  model_input(encoder->feed_forward, norm_output->data);
  model_forward(encoder->feed_forward);

  float *ff_output = MODEL_OUTPUT(encoder->feed_forward);

  Iray1D *o = iray1d_alloc(encoder->embed_dim);
  for (size_t i = 0; i < o->rows; i++) {
    o->data[i] = ff_output[i];
  }

  iray1d_free(norm_output);
  iray1d_free(flatted);
  iray2d_free(output);
  free(ff_output);

  return o;
}
void transformers_encoder_free(Encoder *encoder) {
  model_free(encoder->feed_forward);
  transformers_mha_free(encoder->mha);
  transformers_norm_free(encoder->norm);
  free(encoder);
}

#endif // !TRANSFORMER_H