#define ICAN_USE_NN
#define ICAN_USE_ARRAY
#include "ican/ican.h"
#include <time.h>

float input[] = {
    0.0f, 0.0f,
    1.0f, 0.0f,
    0.0f, 1.0f,
    1.0f, 1.0f
};

float target[] = {
    0.0f,
    0.0f,
    0.0f,
    1.0f
};

int main() {
    srand(time(0));
    Model *model = model_alloc(2);
    Compiler *compiler = compiler_alloc(FiniteDiff, WarmupDecay);
    Layer *layer = layer_dense(2, 1);
    Layer *layer2 = layer_activation(Sigmoid);


    model_add(model, layer);
    model_add(model, layer2);
    model_randomize(model, RandomNormal);
    model_compile(model, compiler);
    compiler->lr = 1e-1;
    compiler->eps = 1e-1;
    model_print(model);

    NNDataSet *input_data = nndataset_1d_from(input, 4, 2);
    NNDataSet *target_data = nndataset_1d_from(target, 4, 1);
    
    model_fit(model, 500, input_data, target_data);

    for (size_t i = 0; i < input_data->size; i++) {
        NNData predict = model_forward(model, input_data->data[i]);
        printf("Target Value = %f ^ Predicted Value = %f\n", target_data->data[i].array1d->data[0], predict.array1d->data[0]);
        nndata_free(predict, 1);
    }

    nndataset_free(input_data);
    nndataset_free(target_data);
    
    model_free(model);
}