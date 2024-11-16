#ifndef OPTIMIZER_H

#define OPTIMIZER_H

#include "ican.h"

void optimizer_finite_diff(Model *model, Iray2D *inputs, Iray2D *outputs, va_list args) {
    float rate = va_arg(args, double);
    float eps = va_arg(args, double);
    if(rate == 0.0) {
        rate = 3e-2;
    }
    if(eps == 0.0) {
        eps = 1e-1;
    }
    for (size_t i = 0; i < inputs->rows; i++)
    {
        float *input = inputs->data[i];
        float *output = outputs->data[i];
        float c = model_cost(model, input, output);
        float saved;
        float g;
        for (size_t l = 0; l < model->layer_count; l++)
        {
            if(model->layers[l]->name != Dense) continue;
            for (size_t j = 0; j < model->layers[l]->inputSize; j++)
            {
                for (size_t k = 0; k < model->layers[l]->outputSize; k++)
                {
                    g = (model_cost(model, input, output) - c) / eps;
                    saved = model->layers[l]->weight->data[j][k];
                    model->layers[l]->weight->data[j][k] += eps;
                    model->layers[l]->weight->data[j][k] = saved;
                    model->layers[l]->weight->data[j][k] -= rate * g;

                    saved = model->layers[l]->bias->data[k];
                    model->layers[l]->bias->data[k] += eps;
                    g = (model_cost(model, input, output) - c) / eps;
                    model->layers[l]->bias->data[k] = saved;
                    model->layers[l]->bias->data[k] -= rate * g;
                }
            }
        }
        print_progress(i, inputs->rows, model_cost(model, inputs->data[i], outputs->data[i]));
    }
}

void optimizer_batch_gradient_descent(Model *model, Iray2D *inputs, Iray2D *outputs, va_list args) {
    float rate = va_arg(args, double);
    if(rate == 0.0) {
        rate = 1e-1;
    }
    for (size_t i = 0; i < inputs->rows; i++)
    {
        float *input = inputs->data[i];
        float *output = outputs->data[i];
        model_input(model, input);
        model_forward(model);
        float *parsed = MODEL_OUTPUT(model);
        bool first = true;
        float **delta = (float **)malloc(sizeof(float) * model->layer_count);
        for (int l = model->layer_count - 1; l >= 0; l--)
        {
            Layer *layer = model->layers[l];
            delta[l] = (float *)malloc(layer->outputSize * sizeof(float));
            if(layer->weight->rows == 0) {
                layer->backward(layer, delta[l], rate);
                continue;
            }
            if(l + 1 <= l){
                for (size_t o = 0; o < layer->output->rows; o++) {
                    layer->output->data[o] = model->layers[l + 1]->input->data[o];
                }
            }
            Layer *prevLayer;
            float *prevDelta;
            for (size_t ll = l + 1; ll < model->layer_count - 1; ll++)
            {
                if(model->layers[ll]->weight->rows != 0){
                    prevLayer = model->layers[ll];
                    prevDelta = delta[ll];
                    break;
                }
            }
            if(first) {
                for (size_t j = 0; j < layer->output->rows; j++)
                {
                    delta[l][j] = (output[j] - parsed[j]) * layer->output->data[j];
                }
                first = false;
            }else {
                for (size_t j = 0; j < layer->outputSize; j++)
                {
                    delta[l][j] = 0.0;
                    for (size_t k = 0; k < prevLayer->outputSize; k++)  
                    {
                        delta[l][j] += prevDelta[k] * prevLayer->weight->data[j][k];
                    }
                    delta[l][j] *= layer->output->data[j];
                }
            }
            layer->backward(layer, delta[l], rate);
        }
        for (size_t l = 0; l < model->layer_count - 1; l++)
        {
            free(delta[l]);
        }
        free(delta);
        print_progress(i, inputs->rows, model_cost(model, inputs->data[i], outputs->data[i]));
    }
}

#endif // !OPTIMIZER_H