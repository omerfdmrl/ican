#include "src/ican.h"
/*

float relu(float x) {
    return fmaxf(0, x);
}

float softmax2(float *input, size_t size, float *output) {
    float max = input[0];
    for (size_t i = 1; i < size; i++) {
        if (input[i] > max) max = input[i];
    }

    float sum = 0;
    for (size_t i = 0; i < size; i++) {
        output[i] = expf(input[i] - max);
        sum += output[i];
    }

    for (size_t i = 0; i < size; i++) {
        output[i] /= sum;
    }

    return sum;
}


typedef struct {
    size_t input_dim;
    size_t output_dim;
    float **weights;
    float *biases;
} LinearLayer;

LinearLayer *linear_alloc(size_t input_dim, size_t output_dim) {
    LinearLayer *layer = malloc(sizeof(LinearLayer));
    layer->input_dim = input_dim;
    layer->output_dim = output_dim;

    layer->weights = malloc(output_dim * sizeof(float *));
    for (size_t i = 0; i < output_dim; i++) {
        layer->weights[i] = malloc(input_dim * sizeof(float));
    }

    layer->biases = calloc(output_dim, sizeof(float));
    return layer;
}

void linear_free(LinearLayer *layer) {
    for (size_t i = 0; i < layer->output_dim; i++) {
        free(layer->weights[i]);
    }
    free(layer->weights);
    free(layer->biases);
    free(layer);
}

void linear_forward(LinearLayer *layer, float *input, float *output) {
    for (size_t i = 0; i < layer->output_dim; i++) {
        output[i] = layer->biases[i];
        for (size_t j = 0; j < layer->input_dim; j++) {
            output[i] += layer->weights[i][j] * input[j];
        }
    }
}


typedef struct {
    size_t embed_dim; // Giriş boyutu (her bir vektörün uzunluğu)
    float *gamma;     // Ölçekleme parametresi (learnable parameter)
    float *beta;      // Kaydırma parametresi (learnable parameter)
} LayerNorm;

LayerNorm *layernorm_alloc(size_t embed_dim) {
    LayerNorm *ln = malloc(sizeof(LayerNorm));
    ln->embed_dim = embed_dim;
    ln->gamma = malloc(embed_dim * sizeof(float));
    ln->beta = malloc(embed_dim * sizeof(float));

    // gamma'yı 1.0 ile başlat, beta'yı 0.0 ile başlat
    for (size_t i = 0; i < embed_dim; i++) {
        ln->gamma[i] = 1.0f;
        ln->beta[i] = 0.0f;
    }

    return ln;
}

void layernorm_free(LayerNorm *ln) {
    if (ln) {
        free(ln->gamma);
        free(ln->beta);
        free(ln);
    }
}

void layernorm_forward(LayerNorm *ln, float *input, size_t seq_len, size_t embed_dim, float *output) {
    for (size_t i = 0; i < seq_len; i++) {
        float *x = &input[i * embed_dim];
        float *y = &output[i * embed_dim];

        // Mean
        float mean = 0.0f;
        for (size_t j = 0; j < embed_dim; j++) {
            mean += x[j];
        }
        mean /= embed_dim;

        // Variance
        float variance = 0.0f;
        for (size_t j = 0; j < embed_dim; j++) {
            variance += (x[j] - mean) * (x[j] - mean);
        }
        variance /= embed_dim;

        // Normalize
        for (size_t j = 0; j < embed_dim; j++) {
            y[j] = (x[j] - mean) / sqrtf(variance + 1e-5f);
            y[j] = y[j] * ln->gamma[j] + ln->beta[j]; // Scale and shift
        }
    }
}


void create_causal_mask(float *mask, size_t seq_len) {
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < seq_len; j++) {
            if (j > i) {
                mask[i * seq_len + j] = -INFINITY; // Gelecek pozisyonlara bakmayı engelle
            } else {
                mask[i * seq_len + j] = 0.0f; // Mevcut ve geçmiş pozisyonlara bakılabilir
            }
        }
    }
}

void masked_attention(float *Q, float *K, float *V, float *mask, size_t seq_len, size_t embed_dim, float *output) {
    float *scores = calloc(seq_len * seq_len, sizeof(float));
    float scale = 1.0f / sqrt(embed_dim);

    // Skorları hesapla: Scores = Q * K^T
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < seq_len; j++) {
            for (size_t k = 0; k < embed_dim; k++) {
                scores[i * seq_len + j] += Q[i * embed_dim + k] * K[j * embed_dim + k];
            }
            scores[i * seq_len + j] *= scale;
            scores[i * seq_len + j] += mask[i * seq_len + j]; // Maskeyi uygula
        }
    }

    // Softmax uygula
    for (size_t i = 0; i < seq_len; i++) {
        float sum = 0.0f;
        for (size_t j = 0; j < seq_len; j++) {
            scores[i * seq_len + j] = exp(scores[i * seq_len + j]);
            sum += scores[i * seq_len + j];
        }
        for (size_t j = 0; j < seq_len; j++) {
            scores[i * seq_len + j] /= sum; // Normalize et
        }
    }

    // Attention Output = Scores * V
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t k = 0; k < embed_dim; k++) {
            for (size_t j = 0; j < seq_len; j++) {
                output[i * embed_dim + k] += scores[i * seq_len + j] * V[j * embed_dim + k];
            }
        }
    }

    free(scores);
}


typedef struct {
    LinearLayer *query;
    LinearLayer *key;
    LinearLayer *value;
    size_t embed_dim;
    size_t num_heads;
} MultiHeadAttention;

MultiHeadAttention *mha_alloc(size_t embed_dim, size_t num_heads) {
    MultiHeadAttention *mha = malloc(sizeof(MultiHeadAttention));
    mha->embed_dim = embed_dim;
    mha->num_heads = num_heads;

    size_t head_dim = embed_dim / num_heads;
    mha->query = linear_alloc(embed_dim, embed_dim);
    mha->key = linear_alloc(embed_dim, embed_dim);
    mha->value = linear_alloc(embed_dim, embed_dim);

    return mha;
}

void mha_free(MultiHeadAttention *mha) {
    linear_free(mha->query);
    linear_free(mha->key);
    linear_free(mha->value);
    free(mha);
}

void mha_forward(MultiHeadAttention *mha, float *input, size_t seq_len, float *output) {
    size_t embed_dim = mha->embed_dim;
    size_t head_dim = embed_dim / mha->num_heads;

    // Allocate temporary buffers
    float *queries = malloc(seq_len * embed_dim * sizeof(float));
    float *keys = malloc(seq_len * embed_dim * sizeof(float));
    float *values = malloc(seq_len * embed_dim * sizeof(float));
    float *scores = malloc(seq_len * seq_len * sizeof(float));
    float *softmaxed = malloc(seq_len * seq_len * sizeof(float));
    float *context = malloc(seq_len * embed_dim * sizeof(float));
    float *masked = malloc(seq_len * seq_len * sizeof(float));

    // Apply query, key, value linear layers
    linear_forward(mha->query, input, queries);
    linear_forward(mha->key, input, keys);
    linear_forward(mha->value, input, values);

    // Compute scaled dot-product attention
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < seq_len; j++) {
            scores[i * seq_len + j] = 0;
            for (size_t k = 0; k < embed_dim; k++) {
                scores[i * seq_len + j] += queries[i * embed_dim + k] * keys[j * embed_dim + k];
            }
            scores[i * seq_len + j] /= sqrtf((float)embed_dim);
        }
    }

    create_causal_mask(masked, seq_len);

    masked_attention(queries, keys, values, masked, seq_len, embed_dim, scores);

    // Apply softmax to scores
    for (size_t i = 0; i < seq_len; i++) {
        softmax2(&scores[i * seq_len], seq_len, &softmaxed[i * seq_len]);
    }

    // Compute context vectors
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < embed_dim; j++) {
            context[i * embed_dim + j] = 0;
            for (size_t k = 0; k < seq_len; k++) {
                context[i * embed_dim + j] += softmaxed[i * seq_len + k] * values[k * embed_dim + j];
            }
        }
    }

    // Write output
    memcpy(output, context, seq_len * embed_dim * sizeof(float));

    // Free temporary buffers
    free(queries);
    free(keys);
    free(values);
    free(scores);
    free(softmaxed);
    free(context);
    free(masked);
}


typedef struct {
    MultiHeadAttention *mha;
    LinearLayer *feedforward1;
    LinearLayer *feedforward2;
    size_t embed_dim;
    LayerNorm *layernorm1;
} TransformerBlock;

TransformerBlock *transformer_block_alloc(size_t embed_dim, size_t num_heads, size_t ff_dim) {
    TransformerBlock *block = malloc(sizeof(TransformerBlock));
    block->embed_dim = embed_dim;
    block->mha = mha_alloc(embed_dim, num_heads);

    block->feedforward1 = linear_alloc(embed_dim, ff_dim);
    block->feedforward2 = linear_alloc(ff_dim, embed_dim);

    block->layernorm1 = layernorm_alloc(embed_dim);

    return block;
}

void transformer_block_free(TransformerBlock *block) {
    mha_free(block->mha);
    linear_free(block->feedforward1);
    linear_free(block->feedforward2);
    layernorm_free(block->layernorm1);
    free(block);
}

void transformer_block_forward(TransformerBlock *block, float *input, size_t seq_len, float *output) {
    // Multi-head attention
    float *mha_output = malloc(seq_len * block->embed_dim * sizeof(float));
    mha_forward(block->mha, input, seq_len, mha_output);

    // Add
    for (size_t i = 0; i < seq_len * block->embed_dim; i++) {
        mha_output[i] += input[i];
    }

    // Norm
    layernorm_forward(block->layernorm1, mha_output, seq_len, block->embed_dim, mha_output);

    // Feedforward network
    float *ff1_output = malloc(seq_len * block->feedforward1->output_dim * sizeof(float));
    linear_forward(block->feedforward1, mha_output, ff1_output);

    for (size_t i = 0; i < seq_len * block->feedforward1->output_dim; i++) {
        ff1_output[i] = relu(ff1_output[i]);
    }

    linear_forward(block->feedforward2, ff1_output, output);

    // Free temporary buffers
    free(mha_output);
    free(ff1_output);
}

*/

int main() {
  // Parametreler
  size_t embed_dim = 512;
  size_t num_heads = 8;
  size_t seq_len = 10; // Örnek dizi uzunluğu
  bool is_mask = true; // Mask kullanımı

  // Multi-Head Attention modeli oluşturuluyor
  MultiHeadAttention *mha = transformers_mha_alloc(embed_dim, num_heads, seq_len, is_mask);

  // Giriş verisi oluşturuluyor (örneğin rasgele)
  for (size_t i = 0; i < num_heads; i++) {
    // Burada giriş verisi doldurulmalı (örnek olarak, rasgele veri)
    for (size_t j = 0; j < seq_len; j++) {
      for (size_t k = 0; k < embed_dim / num_heads; k++) {
        mha->sdpa[i]->query->data[j][k] = (float)rand() / (float)RAND_MAX; // Query için rastgele değer
        mha->sdpa[i]->key->data[j][k] = (float)rand() / (float)RAND_MAX;   // Key için rastgele değer
        mha->sdpa[i]->value->data[j][k] = (float)rand() / (float)RAND_MAX; // Value için rastgele değer
      }
    }
  }

  // MHA forward işlemi
  Iray2D *output = transformers_mha_forward(mha);

  // Çıktıyı kontrol et (örneğin, ilk 5 değeri yazdır)
  printf("Output (first 5 values):\n");
  for (size_t i = 0; i < 5; i++) {
    printf("%f ", output->data[i][0]);  // Çıktının ilk 5 değerini yazdır
  }
  printf("\n");

  // Belleği serbest bırak
  transformers_mha_free(mha);
  iray2d_free(output);

  return 0;
}