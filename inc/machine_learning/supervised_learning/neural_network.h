#ifndef ICAN_MACHINELEARNING_SUPERVISEDLEARNING_NEURALNETWORK_H

#define ICAN_MACHINELEARNING_SUPERVISEDLEARNING_NEURALNETWORK_H

#define ICAN_USE_ARRAY

#include "ican.h"
#include <time.h>

union u_nn_data {
    Array1D *array1d;
    Array2D *array2d;
};

enum e_layer_name {
    Dense,
    Activation,
    Dropout,
    Shuffle,
    Flatten,
    Input,
    Output,
    BatchNormalization
};

enum e_layer_shape {
    array1d = 1,
    array2d = 2
};

static const char* LayerNamesChar[] = {
	[Dense] = "Dense",
	[Activation] = "Activation",
	[Dropout] = "Dropout",
	[Shuffle] = "Shuffle",
	[Flatten] = "Flatten",
	[Input] = "Input",
	[Output] = "Output",
    [BatchNormalization] = "BatchNormalization"
};
#define LAYER_NAME(name) LayerNamesChar[(name)]

enum e_randomizer {
    Ones,
    Zeros,
    RandomHeUniform,
    RandomNormal,
    RandomUniform,
    RandomXavier,
};

enum e_optimizer {
    FiniteDiff,
    BatchGradientDescent,
    MiniBatchGradientDescent,
    StochasticGradientDescent,
    RMSprop,
    Adam
};

enum e_activation_type {
	Sigmoid,
    Tanh,
    Softmax,
    Relu
};

enum e_compiler_schedule {
    WarmupDecay,
    CosineAnnealing
};

struct s_nn_data_set {
    union u_nn_data *data;
    int32 size;
    enum e_layer_shape shape;
};

struct s_compiler {
    enum e_optimizer optimizer;
    enum e_compiler_schedule schedule;
    float lr;
    float lr_min;
    float lr_max;
    float eps;
    int16 warmup_steps;
    int32 d;
    int32 e;
    void (*update)(struct s_compiler *compiler, int t);
};

struct s_layer {
    enum e_layer_name name;
    int64 input_size;
    int64 output_size;
    Array2D *weight;
    Array1D *bias;
    Array1D *param;
    union u_nn_data (*forward)(struct s_layer *layer, union u_nn_data);
	union u_nn_data (*backward)(struct s_layer *layer, union u_nn_data, Array1D *delta, float rate);
    enum e_layer_shape shape;
};

struct s_model {
    int16 layer_count;
    struct s_layer **layers;
    struct s_compiler *compiler;
};

typedef union u_nn_data NNData;
typedef enum e_layer_name LayerName;
typedef enum e_layer_shape LayerShape;
typedef enum e_randomizer Randomizer;
typedef enum e_optimizer Optimizer;
typedef enum e_activation_type ActivationType;
typedef enum e_compiler_schedule CompilerSchedule;
typedef struct s_nn_data_set NNDataSet;
typedef struct s_compiler Compiler;
typedef struct s_layer Layer;
typedef struct s_model Model;

NNData nndata_alloc(Array1D *array1d, Array2D *array2d);
void nndata_free(NNData data, int shape);
NNDataSet *nndataset_1d_from(float *data, int64 size, int64 rows);
void nndataset_free(NNDataSet *dataset);

Layer *layer_alloc(LayerName name, LayerShape shape, int64 input_size, int64 output_size, int64 param_size, NNData (*forward)(Layer *layer, NNData input), NNData (*backward)(Layer *layer, NNData input, Array1D *delta, float rate));
Layer *layer_dense(int64 input_size, int64 output_size);
Layer *layer_activation(ActivationType activation);
Layer *layer_dropout(float rate);
Layer *layer_shuffle(float rate);
void layer_free(Layer *layer);
void layer_print(Layer *layer);

Compiler *compiler_alloc(Optimizer optimizer, CompilerSchedule schedule);
void compiler_update(Compiler *compiler, int t);
void compiler_free(Compiler *compiler);

Model *model_alloc(int16 layer_count);
void model_add(Model *model, Layer *layer);
NNData model_forward(Model *model, NNData input);
float model_cost(Model *model, NNDataSet *inputs, NNDataSet *outputs);
void model_randomize(Model *model, Randomizer randomizer, ...);
void model_compile(Model *model, Compiler *compiler);
void model_fit(Model *model, int32 epochs, NNDataSet *inputs, NNDataSet *outputs);
void model_free(Model *model);
void model_print(Model *model);

#endif // !ICAN_MACHINELEARNING_SUPERVISEDLEARNING_NEURALNETWORK_H