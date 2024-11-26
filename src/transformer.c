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

void make_dropout(Iray2D *data, float rate) {
  for (size_t i = 0; i < data->rows; i++) {
    for (size_t j = 0; j < data->cols; j++) {
      float rand = random_uniform(0, 1);
      if(rand < rate) {
          data->data[i][j] = 0;    
      }
    }
  }
}

TransformerScaledDotProductAttention *transformer_sdpa_alloc(size_t seq_len, size_t head_dim, size_t embed_dim, bool is_mask) {
  TransformerScaledDotProductAttention *sdpa = (TransformerScaledDotProductAttention *)malloc(sizeof(TransformerScaledDotProductAttention));
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
void transformer_sdpa_free(TransformerScaledDotProductAttention *sdpa) {
  iray2d_free(sdpa->query);
  iray2d_free(sdpa->key);
  iray2d_free(sdpa->value);
  iray2d_free(sdpa->W_q);
  iray2d_free(sdpa->W_k);
  iray2d_free(sdpa->W_v);
  free(sdpa);
}
Iray2D *transformer_sdpa_forward(TransformerScaledDotProductAttention *sdpa) {
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

TransformerMultiHeadAttention *transformer_mha_alloc(size_t embed_dim, size_t num_heads, size_t seq_len, bool is_mask) {
  TransformerMultiHeadAttention *mha = (TransformerMultiHeadAttention *)malloc(sizeof(TransformerMultiHeadAttention));
  mha->sdpa = (TransformerScaledDotProductAttention **)malloc(sizeof(TransformerScaledDotProductAttention *) * num_heads);
  mha->embed_dim = embed_dim;
  mha->num_heads = num_heads;
  mha->seq_len = seq_len;
  mha->W_o = iray2d_alloc(embed_dim * num_heads, embed_dim);

  for (size_t i = 0; i < num_heads; i++) {
    mha->sdpa[i] = transformer_sdpa_alloc(seq_len, embed_dim / num_heads, embed_dim, is_mask);
  }
  return mha;
}
void transformer_mha_free(TransformerMultiHeadAttention *mha) {
  for (size_t i = 0; i < mha->num_heads; i++) {
    transformer_sdpa_free(mha->sdpa[i]);
  }
  iray2d_free(mha->W_o);
  free(mha->sdpa);
  free(mha);
}
Iray2D *transformer_mha_forward(TransformerMultiHeadAttention *mha) {
  Iray2D **sdpa_outputs = (Iray2D **)malloc(sizeof(Iray2D *) * mha->num_heads);
  for (size_t i = 0; i < mha->num_heads; i++) {
    mha->sdpa[i]->query = iray2d_dot(mha->sdpa[i]->query, mha->sdpa[i]->W_q);
    mha->sdpa[i]->key = iray2d_dot(mha->sdpa[i]->key, mha->sdpa[i]->W_k);
    mha->sdpa[i]->value = iray2d_dot(mha->sdpa[i]->value, mha->sdpa[i]->W_v);
    sdpa_outputs[i] = transformer_sdpa_forward(mha->sdpa[i]);
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

TransformerNorm *transformer_norm_alloc(size_t embed_dim) {
  TransformerNorm *norm = malloc(sizeof(TransformerNorm));
  norm->embed_dim = embed_dim;
  norm->gamma = malloc(embed_dim * sizeof(float));
  norm->beta = malloc(embed_dim * sizeof(float));

  for (size_t i = 0; i < embed_dim; i++) {
      norm->gamma[i] = 1.0f;
      norm->beta[i] = 0.0f;
  }

  return norm;
}
void transformer_norm_free(TransformerNorm *norm) {
  free(norm->gamma);
  free(norm->beta);
  free(norm);
}
Iray2D *transformer_norm_forward(TransformerNorm *ln, Iray2D *input, size_t seq_len, size_t embed_dim) {
  Iray2D *output = iray2d_alloc(seq_len, embed_dim);

  for (size_t i = 0; i < seq_len; i++) {
    float mean = 0.0f;
    for (size_t j = 0; j < embed_dim; j++) {
        mean += input->data[i][j];
    }
    mean /= embed_dim;

    float variance = 0.0f;
    for (size_t j = 0; j < embed_dim; j++) {
        variance += (input->data[i][j] - mean) * (input->data[i][j] - mean);
    }
    variance /= embed_dim;

    for (size_t j = 0; j < embed_dim; j++) {
        output->data[i][j] = (input->data[i][j] - mean) / sqrtf(variance + 1e-3f);
        output->data[i][j] = output->data[i][j] * ln->gamma[j] + ln->beta[j];
    }
  }

  return output;
}

TransformerEncoderBlock *transformer_encoder_block_alloc(size_t embed_dim, size_t num_heads, size_t seq_len, float dropout, bool is_mask) {
  TransformerEncoderBlock *encoder = (TransformerEncoderBlock *)malloc(sizeof(TransformerEncoderBlock));
  encoder->mha = transformer_mha_alloc(embed_dim, num_heads, seq_len, is_mask);
  encoder->norm = transformer_norm_alloc(embed_dim);
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
  encoder->dropout = dropout;
  model_randomize(RandomNormal, encoder->feed_forward);
  return encoder;
}
Iray2D *transformer_encoder_block_forward(TransformerEncoderBlock *encoder, Iray2D *input) {
  ASSERT_MSG(input->rows == encoder->seq_len && input->cols == encoder->embed_dim, "Input dimensions do not match encoder configuration");

  for (size_t i = 0; i < encoder->num_heads; i++) {
      encoder->mha->sdpa[i]->query = input;
      encoder->mha->sdpa[i]->key = input;
      encoder->mha->sdpa[i]->value = input;
  }
  Iray2D *output = transformer_mha_forward(encoder->mha);

  for (size_t i = 0; i < encoder->seq_len; i++) {
      for (size_t j = 0; j < encoder->embed_dim; j++) {
          output->data[i][j] += input->data[i][j];
      }
  }

  Iray2D *norm_output = transformer_norm_forward(encoder->norm, output, encoder->seq_len, encoder->embed_dim);
  make_dropout(norm_output, encoder->dropout);

  Iray1D *norm_o_flatted = iray2d_flatten(norm_output);

  model_input(encoder->feed_forward, norm_o_flatted->data);
  model_forward(encoder->feed_forward);

  float *ff_output = MODEL_OUTPUT(encoder->feed_forward);

  Iray2D *ff_layer_output = iray2d_alloc(encoder->seq_len, encoder->embed_dim);
  for (size_t i = 0; i < encoder->seq_len; i++) {
      model_input(encoder->feed_forward, norm_output->data[i]);
      model_forward(encoder->feed_forward);

      float *ff_output = MODEL_OUTPUT(encoder->feed_forward);
      for (size_t j = 0; j < encoder->embed_dim; j++) {
          ff_layer_output->data[i][j] = ff_output[j];
      }
  }

  for (size_t i = 0; i < encoder->seq_len; i++) {
      for (size_t j = 0; j < encoder->embed_dim; j++) {
          ff_layer_output->data[i][j] += norm_output->data[i][j];
      }
  }

  Iray2D *norm2_output = transformer_norm_forward(encoder->norm, ff_layer_output, encoder->seq_len, encoder->embed_dim);
  make_dropout(norm2_output, encoder->dropout);

  iray2d_free(norm_output);
  iray1d_free(norm_o_flatted);
  iray2d_free(output);
  iray2d_free(ff_layer_output);
  free(ff_output);

  return norm2_output;
}
void transformer_encoder_block_free(TransformerEncoderBlock *encoder) {
  model_free(encoder->feed_forward);
  transformer_mha_free(encoder->mha);
  transformer_norm_free(encoder->norm);
  free(encoder);
}

TransformerEmbedding *transformer_embedding_alloc(size_t vocab_size, size_t embed_dim) {
  TransformerEmbedding *embedding = (TransformerEmbedding *)malloc(sizeof(TransformerEmbedding));
  embedding->embed_dim = embed_dim;
  embedding->vocab_size = vocab_size;
  embedding->W = iray2d_alloc(vocab_size, embed_dim);
  /** Will Delete */
  for (size_t i = 0; i < vocab_size; i++) {
    for (size_t j = 0; j < embed_dim; j++) {
      embedding->W->data[i][j] = random_uniform(0, 1);
    }
  }
  return embedding;
}
void transformer_embedding_free(TransformerEmbedding *embedding) {
  iray2d_free(embedding->W);
  free(embedding);
}
Iray2D *transformer_embedding_forward(TransformerEmbedding *embedding, Iray2D *input) {
  Iray2D *output = iray2d_alloc(input->rows, input->cols);
  for (size_t i = 0; i < output->rows; i++) {
    for (size_t j = 0; j < output->cols; j++) {
      output->data[i][j] = embedding->W->data[i][j];
    }
  }
  return output;
}

TransformerPositionalEncoding *transformer_positional_encoding_alloc(size_t seq_len, size_t embed_dim) {
  TransformerPositionalEncoding *pe = (TransformerPositionalEncoding *)malloc(sizeof(TransformerPositionalEncoding));
  pe->embed_dim = embed_dim;
  pe->seq_len = seq_len;
  pe->encoding = iray2d_alloc(seq_len, embed_dim);
  for (size_t i = 0; i < seq_len; i++) {
    for (size_t j = 0; j < embed_dim; j++) {
      float denominator = (float)i / pow(10000.0, (2 * j) / (float)embed_dim);
      if (j % 2 == 0) {
        pe->encoding->data[i][j] = sin(denominator);
      }else {
        pe->encoding->data[i][j] = cos(denominator);
      }
    }
  }
  
  return pe;
}
void transformer_positional_encoding_free(TransformerPositionalEncoding *pe) {
  free(pe);
}
Iray2D *transformer_positional_encoding_forward(TransformerPositionalEncoding *pe, Iray2D *input) {
  Iray2D *output = iray2d_alloc(input->rows, input->cols);
  for (size_t i = 0; i < output->rows; i++) {
    for (size_t j = 0; j < output->cols; j++) {
      output->data[i][j] = input->data[i][j] + pe->encoding->data[i][j];
    }
  }
  return output;
}

TransformerEncoder *transformer_encoder_alloc(size_t seq_len, size_t vocab_size, size_t embed_dim, size_t num_heads, size_t num_blocks, float dropout, bool is_mask) {
  TransformerEncoder *encoder = (TransformerEncoder *)malloc(sizeof(TransformerEncoder));
  encoder->positional_encoding = transformer_positional_encoding_alloc(seq_len, embed_dim);
  encoder->embedding = transformer_embedding_alloc(vocab_size, embed_dim);
  encoder->blocks = (TransformerEncoderBlock **)malloc(sizeof(TransformerEncoderBlock *) * num_blocks);
  for (size_t i = 0; i < num_blocks; i++) {
    encoder->blocks[i] = transformer_encoder_block_alloc(embed_dim, num_heads, seq_len, dropout, is_mask);
  }
  encoder->seq_len = seq_len;
  encoder->vocab_size = vocab_size;
  encoder->embed_dim = embed_dim;
  encoder->num_heads = num_heads;
  encoder->num_blocks = num_blocks;
  encoder->dropout = dropout;
  encoder->is_mask = is_mask;
  return encoder;
}
void transformer_encoder_free(TransformerEncoder *encoder) {
  transformer_positional_encoding_free(encoder->positional_encoding);
  transformer_embedding_free(encoder->embedding);
  for (size_t i = 0; i < encoder->num_blocks; i++) {
    transformer_encoder_block_free(encoder->blocks[i]);
  }
  free(encoder->blocks);
  free(encoder);
}
Iray2D *transformer_encoder_forward(TransformerEncoder *encoder, Iray2D *input) {
  Iray2D *embedding_output = transformer_embedding_forward(encoder->embedding, input);
  Iray2D *pe_output = transformer_positional_encoding_forward(encoder->positional_encoding, embedding_output);
  for (size_t i = 0; i < encoder->num_blocks; i++) {
    pe_output = transformer_encoder_block_forward(encoder->blocks[i], pe_output);
  }
  iray2d_free(embedding_output);
  return pe_output;
}

#endif // !TRANSFORMER_H