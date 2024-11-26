#ifndef ICAN_H

#define ICAN_H

#include <stddef.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdbool.h>
#include <png.h>
#include <jpeglib.h>
#include <sys/types.h>
#include <dirent.h>
#include <SDL2/SDL.h>
#include <cjson/cJSON.h>

#define PI 3.14159265358979323846

typedef enum{
    LOG_LEVEL_FATAL = 0,
    LOG_LEVEL_ERROR = 1,
    LOG_LEVEL_WARN = 2,
    LOG_LEVEL_INFO = 3,
    LOG_LEVEL_DEBUG = 4,
    LOG_LEVEL_TRACE = 5
} ILogLevels;

#define ASSERT(expr)                                         \
    {                                                                \
        if (expr) {                                                  \
        } else {                                                     \
            report_assertion_failure(#expr, "", __FILE__, __LINE__); \
            __builtin_trap();                                        \
        }                                                            \
    }

#define ASSERT_MSG(expr, message)                                 \
    {                                                                     \
        if (expr) {                                                       \
        } else {                                                          \
            report_assertion_failure(#expr, message, __FILE__, __LINE__); \
            __builtin_trap();                                             \
        }                                                                 \
    }

typedef struct Iray1D {
    size_t rows;
    float *data;
} Iray1D;
typedef struct Iray2D {
    size_t rows;
    size_t cols;
    float **data;
} Iray2D;
typedef struct Iray3D {
    size_t rows;
    size_t cols;
    size_t depth;
    float ***data;
} Iray3D;

typedef enum {
    IMG_RGBA,
    IMG_RGB,
    IMG_GRAY
} ImgTypes;
static const int ImgTypesSize[] = {
    [IMG_RGBA] = 4,
    [IMG_RGB] = 3,
    [IMG_GRAY] = 1,
};

typedef struct {
    SDL_Window *window;
    SDL_Renderer *renderer;
} ISDLContext;

extern float CONV_EMBOSS[3][3];
extern float CONV_EMBOSS2[3][3];
extern float CONV_LAPLACIAN[3][3];
extern float CONV_BLUR[3][3];
extern float CONV_SOBELX[3][3];
extern float CONV_SOBELY[3][3];

typedef enum {
    PAD_POST,
    PAD_PRE
} PadTypes;

typedef struct {
    cJSON *vocab;
    size_t  vocab_size;
    cJSON *added_tokens;
    cJSON *bpe_rules;
    size_t bpe_rule_size;
    char *unk_token;
    char *bos_token;
    char *eos_token;
    size_t max_length;
} Tokenizer;

typedef enum {
	Dense,
    RNN,
    GRU,
	Activation,
	Dropout,
	Shuffle,
    BatchNormalization
} LayerNames;

static const char* LayerNamesChar[] = {
	[Dense] = "Dense",
	[RNN] = "RNN",
	[GRU] = "GRU",
	[Activation] = "Activation",
	[Dropout] = "Dropout",
	[Shuffle] = "Shuffle",
    [BatchNormalization] = "BatchNormalization"
};
#define LAYER_NAME(name) LayerNamesChar[(name)]

typedef struct Layer Layer;

struct Layer {
	LayerNames name;
	size_t inputSize;
	size_t outputSize;
	Iray1D *input;
	Iray1D *output;
	Iray1D *bias;
	Iray1D *params;
	Iray2D *weight;
	void (*forward)(Layer *layer);
	void (*backward)(Layer *layer, float *delta, float rate);
};

typedef enum {
	Sigmoid,
    Tanh,
    Softmax,
    RELU
} ActivationTypes;

typedef struct Model Model;
struct Model
{
	size_t layer_count;
	Layer **layers;
	void (*step)(Iray1D *loss_history);
};

typedef enum
{
	FiniteDiff,
	BatchGradientDescent,
} Itimizers;
typedef enum
{
	Zeros,
	Ones,
	RandomUniform,
	RandomNormal,
	RandomXavier,
	RandomHeUniform
} Initializers;

#define MODEL_OUTPUT(m) m->layers[m->layer_count - 1]->output->data

void LOG(ILogLevels level, const char* message, ...);
void report_assertion_failure(const char* expression, const char* message, const char* file, int line);

#define MAX_LINE_BYTE 2080

typedef struct TransformerScaledDotProductAttention {
    Iray2D *query;
    Iray2D *key;
    Iray2D *value;
    Iray2D *W_q;
    Iray2D *W_k;
    Iray2D *W_v;
    size_t head_dim;
    size_t embed_dim;
    size_t seq_len;
    bool is_mask;
} TransformerScaledDotProductAttention;

typedef struct TransformerMultiHeadAttention {
    TransformerScaledDotProductAttention **sdpa;
    Iray2D *W_o;
    size_t embed_dim;
    size_t seq_len;
    size_t num_heads;
} TransformerMultiHeadAttention;

typedef struct TransformerNorm {
    size_t embed_dim;
    float *gamma;
    float *beta;
} TransformerNorm;

typedef struct TransformerEncoderBlock {
    TransformerMultiHeadAttention *mha;
    TransformerNorm *norm;
    Model *feed_forward;
    size_t num_heads;
    size_t embed_dim;
    size_t seq_len;
    float dropout;
} TransformerEncoderBlock;

typedef struct TransformerEmbedding {
    size_t vocab_size;
    size_t embed_dim;
    Iray2D *W;
} TransformerEmbedding;

typedef struct TransformerPositionalEncoding {
    size_t seq_len;
    size_t embed_dim;
    Iray2D *encoding;
} TransformerPositionalEncoding;

typedef struct TransformerEncoder {
    TransformerEncoderBlock **blocks;
    TransformerEmbedding *embedding;
    TransformerPositionalEncoding *positional_encoding;
    size_t seq_len;
    size_t vocab_size;
    size_t embed_dim;
    size_t num_heads;
    size_t num_blocks;
    float dropout;
    bool is_mask;
} TransformerEncoder;

TransformerScaledDotProductAttention *transformer_sdpa_alloc(size_t seq_len, size_t head_dim, size_t embed_dim, bool is_mask);
void transformer_sdpa_free(TransformerScaledDotProductAttention *sdpa);
Iray2D *transformer_sdpa_forward(TransformerScaledDotProductAttention *sdpa);
TransformerMultiHeadAttention *transformer_mha_alloc(size_t embed_dim, size_t num_heads, size_t seq_len, bool is_mask);
void transformer_mha_free(TransformerMultiHeadAttention *mha);
Iray2D *transformer_mha_forward(TransformerMultiHeadAttention *mha);
TransformerNorm *transformer_norm_alloc(size_t embed_dim);
void transformer_norm_free(TransformerNorm *norm);
Iray2D *transformer_norm_forward(TransformerNorm *ln, Iray2D *input, size_t seq_len, size_t embed_dim);
TransformerEncoderBlock *transformer_encoder_block_alloc(size_t embed_dim, size_t num_heads, size_t seq_len, float dropout, bool is_mask);
Iray2D *transformer_encoder_block_forward(TransformerEncoderBlock *encoder, Iray2D *input);
void transformer_encoder_block_free(TransformerEncoderBlock *encoder);
TransformerEmbedding *transformer_embedding_alloc(size_t vocab_size, size_t embed_dim);
void transformer_embedding_free(TransformerEmbedding *embedding);
Iray2D *transformer_embedding_forward(TransformerEmbedding *embedding, Iray2D *input);
TransformerPositionalEncoding *transformer_positional_encoding_alloc(size_t seq_len, size_t embed_dim);
void transformer_positional_encoding_free(TransformerPositionalEncoding *pe);
Iray2D *transformer_positional_encoding_forward(TransformerPositionalEncoding *pe, Iray2D *input);
TransformerEncoder *transformer_encoder_alloc(size_t seq_len, size_t vocab_size, size_t embed_dim, size_t num_heads, size_t num_blocks, float dropout, bool is_mask);
void transformer_encoder_free(TransformerEncoder *encoder);
Iray2D *transformer_encoder_forward(TransformerEncoder *encoder, Iray2D *input);

Iray1D *iray1d_alloc(size_t rows);
void iray1d_free(Iray1D *iray);
Iray2D *iray2d_alloc(size_t rows, size_t cols);
void iray2d_free(Iray2D *iray);
Iray3D *iray3d_alloc(size_t rows, size_t cols, size_t depth);
void iray3d_free(Iray3D *iray);
Iray1D *iray1d_add(Iray1D *A, Iray1D *B);
Iray1D *iray1d_dot(Iray1D *A, Iray1D *B);
Iray1D *iray1d_scale(Iray1D *A, float scale);
Iray1D *iray1d_slice(Iray1D *iray, size_t start, size_t end);
Iray1D *iray1d_apply(Iray1D *iray1d, float(*fn)(float x));
Iray1D *iray1d_fill(Iray1D *iray, float value);
float iray1d_max(Iray1D *iray);
Iray1D *iray1d_clone(Iray1D *iray);
void iray1d_print(Iray1D *iray);
Iray2D *iray2d_transpose(Iray2D *iray);
Iray2D *iray2d_dot(Iray2D *A, Iray2D *B);
Iray2D *iray2d_div(Iray2D *A, Iray2D *B);
Iray2D *iray2d_scale(Iray2D *A, float scale);
Iray2D *iray2d_softmax(Iray2D *matrix, int axis);
Iray2D *iray2d_concat(Iray2D **matrices, size_t num_matrices);
Iray2D *iray2d_slice(Iray2D *iray, size_t start, size_t end);
Iray2D *iray2d_add(Iray2D *A, Iray2D *B);
Iray2D *iray2d_apply(Iray2D *iray, float(*fn)(float value));
Iray2D *iray2d_fill(Iray2D *iray, float value);
float iray2d_max(Iray2D *iray);
Iray2D *iray2d_clone(Iray2D *iray);
Iray1D *iray2d_flatten(Iray2D *iray);
void iray2d_print(Iray2D *iray);
Iray3D *iray3d_add(Iray3D *A, Iray3D *B);
Iray3D *iray3d_apply(Iray3D *iray, float (*fn)(float value));
Iray3D *iray3d_fill(Iray3D *iray, float value);
float iray3d_max(Iray3D *iray);
Iray3D *iray3d_clone(Iray3D *iray);
Iray3D *iray3d_transpose(Iray3D *iray);
void iray3d_print(Iray3D *iray);

Iray3D *img_read_png(FILE *fp, ImgTypes type);
Iray3D *img_read_jpg(FILE *fp, ImgTypes type);
Iray3D *img_read(const char *imageName, ImgTypes type);
Iray3D *img_resize(Iray3D *img, size_t width, size_t height);
Iray3D *img_rotate(Iray3D *img, float angle);
Iray3D **imgs_read(const char *folderPath, size_t count, ImgTypes type);
Iray3D **imgs_read_wc(const char *folderPath, size_t count, ImgTypes type, Iray3D *(*callback)(Iray3D *img));
void imgs_free(Iray3D **imgs, size_t count);
void img_write_png(FILE *fp, Iray3D *img);
void img_write_jpg(FILE *fp, Iray3D *img);
void img_write(const char *imageName, Iray3D *img);

ISDLContext *isdl_alloc(int width, int height, bool resizable);
void isdl_free(ISDLContext *context);
void img_show(ISDLContext *context, Iray3D *img);
void model_show(ISDLContext *context, Model *model, Iray1D *loss_history);

Iray3D *img_conv(Iray3D *img, float kernel[3][3], size_t stride);
Iray3D *img_max_pool(Iray3D* img, size_t pool_size);
Iray3D* img_min_pool(Iray3D* img, size_t pool_size);
Iray3D *img_mean_pool(Iray3D* img, size_t pool_size);
Iray3D *img_edge_detect(Iray3D *img);

Iray2D *one_hot_encoding(Iray2D *data);
Iray2D *standard_scaler(Iray2D *data);
Iray2D *pad_sequences(Iray2D *data, size_t maxLength, PadTypes padding, PadTypes truncating, float value);
Tokenizer *tokenizer_alloc();
void tokenizer_free(Tokenizer *tokenizer);
void load_vocab_from_json(Tokenizer *tokenizer, const char *file_path);
void load_bpe_from_text(Tokenizer *tokenizer, const char *file_path);
void load_tokenizer_config_from_json(Tokenizer *tokenizer, const char *file_path);
void load_tokenizer_from_json(Tokenizer *tokenizer, const char *file_path);
bool match_token_with_properties(const char *token, cJSON *properties);
Iray1D *fits_on_text(Tokenizer *tokenizer, const char *text);

float sigmoid(float x);
float dsigmoid(float x);
float dtanh(float x);
float relu(float x);
float random_uniform(float low,float high);
float random_normal(float mean, float stddev);
float random_xavier(float fan_in, float fan_out);
float random_xavier_sqrt(float fan_in, float fan_out);
float random_xavier_rand(float a);
float random_heuniform(float fan_in, float fan_out);
void print_progress_header(size_t epoch, size_t total_epochs, float cost);
void print_progress(size_t count, size_t max, float cost);
void print_progress_footer(time_t timer, float cost);
Iray2D *csv_read(const char *filename, size_t rows, size_t cols);
Iray2D *bin_read(const char *filename, size_t rows, size_t cols);

void layer_free(Layer *layer);
void layer_print(Layer *layer);
Layer *layer_dense(size_t inputSize, size_t outputSize);
Layer *layer_rnn(size_t inputSize, size_t outputSize);
Layer *layer_gru(size_t inputSize, size_t outputSize);
Layer *layer_activation(ActivationTypes activation);
Layer *layer_dropout(float rate);
Layer *layer_shuffle(float rate);
Layer *layer_batch_normalization(size_t inputSize);

void initializer_ones(Model *model);
void initializer_random_heuniform(Model *model);
void initializer_random_normal(Model *model, va_list args);
void initializer_random_uniform(Model *model, va_list args);
void initializer_random_xavier(Model *model);
void initializer_zeros(Model *model);

void optimizer_finite_diff(Model *model, Iray2D *inputs, Iray2D *outputs, va_list args);
void optimizer_batch_gradient_descent(Model *model, Iray2D *inputs, Iray2D *outputs, va_list args);

Model *model_alloc(size_t layerCount);
void model_add(Model *model, Layer *layer);
void model_randomize(Initializers initializer, Model *model, ...);
void model_forward(Model *model);
void model_input(Model *model, float *input);
float model_cost(Model *model, float *input, float *output);
Iray1D *model_learn(Itimizers optimizer, Model *model, size_t epoch, Iray2D *inputs, Iray2D *outputs, ...);
void model_print(Model *model);
void model_free(Model *model);

void io_export(Model *model, const char *fileName);
Model *io_import(const char *fileName);

#endif // !ICAN_H