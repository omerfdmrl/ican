#ifndef ITIMIZER_H

#define ITIMIZER_H

#include "ican.h"

void itimizer_finite_diff(Model *model, Iray2D *inputs, Iray2D *outputs, va_list args) {
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
                    saved = model->layers[l]->weight->data[j][k];
                    model->layers[l]->weight->data[j][k] += eps;
                    g = (model_cost(model, input, output) - c) / eps;
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

void itimizer_batch_gradient_descent(Model *model, Iray2D *inputs, Iray2D *outputs, va_list args) {
    float rate = va_arg(args, double);
    if(rate == 0.0) {
        rate = 1e-1;
    }

    size_t time_steps = inputs->rows;
    for (size_t t = 0; t < time_steps; t++) {
        float *input = inputs->data[t];
        float *output = outputs->data[t];

        model_input(model, input);
        model_forward(model);

        float *parsed = MODEL_OUTPUT(model);
        bool first = true;

        float **delta = (float **)malloc(sizeof(float) * model->layer_count);

        for (int l = model->layer_count - 1; l >= 0; l--) {
            Layer *layer = model->layers[l];
            delta[l] = (float *)malloc(layer->outputSize * sizeof(float));

            if (layer->name == Activation) {
                layer->backward(layer);
                continue;
            }

            if (layer->name == RNN) {
                if (first) {
                    for (size_t j = 0; j < layer->outputSize; j++) {
                        delta[l][j] = (output[j] - parsed[j]) * dtanh(layer->output->data[j]);
                    }
                    first = false;
                } else {
                    Layer *prevLayer = NULL;
                    float *prevDelta = NULL;

                    for (int ll = l + 1; ll < model->layer_count - 1; ll++) {
                        if (model->layers[ll]->name != Activation) {
                            prevLayer = model->layers[ll];
                            prevDelta = delta[ll];
                            break;
                        }
                    }

                    for (size_t j = 0; j < layer->outputSize; j++) {
                        delta[l][j] = 0.0;
                        for (size_t k = 0; k < prevLayer->outputSize; k++) {
                            delta[l][j] += prevDelta[k] * prevLayer->weight->data[k][j];
                        }
                        delta[l][j] *= dtanh(layer->output->data[j]);
                    }
                }

                for (size_t j = 0; j < layer->inputSize; j++) {
                    for (size_t k = 0; k < layer->outputSize; k++) {
                        float dw = rate * delta[l][k] * layer->input->data[j];
                        layer->weight->data[j][k] += dw;
                        if (layer->name == GRU) {
                            layer->weight->data[layer->inputSize][k] += dw;
                            layer->weight->data[layer->inputSize + 1][k] += dw;
                            layer->weight->data[layer->inputSize + 2][k] += dw;
                        }
                    }
                }

                for (size_t j = 0; j < layer->outputSize; j++) {
                    float db = delta[l][j] * rate;
                    layer->bias->data[j] += db;
                    if (layer->name == GRU) {
                        layer->bias->data[layer->outputSize + j] += db;
                        layer->bias->data[layer->outputSize*2 + j] += db;
                    }
                }
            } else {
                if (first) {
                    for (size_t j = 0; j < layer->output->rows; j++) {
                        delta[l][j] = (output[j] - parsed[j]) * layer->output->data[j];
                    }
                    first = false;
                } else {
                    Layer *prevLayer = NULL;
                    float *prevDelta = NULL;
                    for (int ll = l + 1; ll < model->layer_count - 1; ll++) {
                        if (model->layers[ll]->name != Activation) {
                            prevLayer = model->layers[ll];
                            prevDelta = delta[ll];
                            break;
                        }
                    }

                    for (size_t j = 0; j < layer->outputSize; j++) {
                        delta[l][j] = 0.0;
                        for (size_t k = 0; k < prevLayer->outputSize; k++) {
                            delta[l][j] += prevDelta[k] * prevLayer->weight->data[k][j];
                        }
                        delta[l][j] *= layer->output->data[j];
                    }
                }

                for (size_t j = 0; j < layer->inputSize; j++) {
                    for (size_t k = 0; k < layer->outputSize; k++) {
                        float dw = rate * delta[l][k] * layer->input->data[j];
                        layer->weight->data[j][k] += dw;
                    }
                }

                for (size_t j = 0; j < layer->outputSize; j++) {
                    float db = delta[l][j] * rate;
                    layer->bias->data[j] += db;
                }
            }
        }

        for (size_t l = 0; l < model->layer_count; l++) {
            if (delta[l] != NULL) {
                free(delta[l]);
            }
        }
        free(delta);

        if (inputs->rows != 1) 
            print_progress(t, inputs->rows, model_cost(model, inputs->data[t], outputs->data[t]));
    }
}


#endif // !ITIMIZER_H