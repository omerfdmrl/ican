#ifndef ILAYER_H

#define ILAYER_H

#include "ican.h"

Layer *layer_alloc(LayerNames name, size_t inputSize, size_t outputSize, size_t paramsSize, void (*forward)(Layer *layer), void (*backward)(Layer *layer, float *delta, float rate)) {
	Layer *layer = malloc(sizeof(Layer));
    layer->name = name;
    layer->inputSize = inputSize;
    layer->outputSize = outputSize;
    layer->input = iray1d_alloc(inputSize);
    layer->output = iray1d_alloc(outputSize);
    layer->forward = forward;
    layer->backward = backward;
    layer->bias = iray1d_alloc(outputSize);
    layer->weight = iray2d_alloc(inputSize, outputSize);
    layer->params = iray1d_alloc(paramsSize);
	return layer;
}

void layer_free(Layer *layer) {
	iray1d_free(layer->bias);
	iray1d_free(layer->params);
	iray1d_free(layer->input);
	iray1d_free(layer->output);
	iray2d_free(layer->weight);
	free(layer);
}

void layer_print(Layer *layer) {
	size_t param = layer->bias->rows + layer->weight->cols * layer->weight->rows + layer->params->rows;
	printf("%-20s(%zu, %zu)%-16s%zu\n", LAYER_NAME(layer->name), layer->inputSize, layer->outputSize,"", param);
}

void layer_dense_forward(Layer *layer) {
    for (size_t i = 0; i < layer->outputSize; i++)
    {
        layer->output->data[i] = 0;
        for (size_t j = 0; j < layer->inputSize; j++) {
            layer->output->data[i] += layer->weight->data[j][i] * layer->input->data[j];
        }
        layer->output->data[i] += layer->bias->data[i];
    }
}

void layer_dense_backward(Layer *layer, float *delta, float rate) {
    for (size_t j = 0; j < layer->inputSize; j++) {
        for (size_t k = 0; k < layer->outputSize; k++) {
            float dw = rate * delta[k] * layer->input->data[j];
            layer->weight->data[j][k] += dw;
        }
    }

    for (size_t j = 0; j < layer->outputSize; j++) {
        float db = delta[j] * rate;
        layer->bias->data[j] += db;
    }
}

void layer_rnn_forward(Layer *layer) {
    for (size_t j = 0; j < layer->outputSize; j++) {
        layer->output->data[j] = 0;
        for (size_t i = 0; i < layer->inputSize; i++) {
            layer->output->data[j] += layer->input->data[i] * layer->weight->data[i][j];
        }
        layer->output->data[j] += layer->params->data[j] * layer->weight->data[layer->inputSize][j];
        layer->output->data[j] += layer->bias->data[j];
        layer->output->data[j] = tanhf(layer->output->data[j]);
        layer->params->data[j] = layer->output->data[j];
    }
}

void layer_rnn_backward(Layer *layer, float *delta, float rate) {
    for (size_t j = 0; j < layer->inputSize; j++) {
        for (size_t k = 0; k < layer->outputSize; k++) {
            float dw = rate * delta[k] * layer->input->data[j];
            layer->weight->data[j][k] += dw;
        }
    }

    for (size_t j = 0; j < layer->outputSize; j++) {
        float db = delta[j] * rate;
        layer->bias->data[j] += db;
    }
}

void layer_gru_forward(Layer *layer) {
    Iray1D *input = layer->input;
    Iray1D *params = layer->params;
    Iray1D *output = layer->output;
    Iray2D *weight = layer->weight;

    Iray1D *bias_reset = iray1d_slice(layer->bias, 0, layer->outputSize);
    Iray1D *bias_update = iray1d_slice(layer->bias, layer->outputSize, layer->outputSize * 2);
    Iray1D *bias_hidden = iray1d_slice(layer->bias, layer->outputSize * 2, layer->outputSize * 3);

    size_t inputSize = layer->inputSize;
    size_t outputSize = layer->outputSize;

    for (size_t j = 0; j < outputSize; j++) {
        output->data[j] = 0;
        float reset_gate = 0;
        float update_gate = 0;

        for (size_t i = 0; i < inputSize; i++) {
            reset_gate += input->data[i] * weight->data[i][j];
            update_gate += input->data[i] * weight->data[inputSize][j];
        }

        reset_gate += bias_reset->data[j];
        update_gate += bias_update->data[j];

        reset_gate = sigmoid(reset_gate);
        update_gate = sigmoid(update_gate);

        float hidden_value = 0;
        for (size_t i = 0; i < inputSize; i++) {
            hidden_value += input->data[i] * weight->data[inputSize + 1][j];
        }

        hidden_value += reset_gate * params->data[j] * weight->data[inputSize + 2][j];
        hidden_value += bias_hidden->data[j];
        hidden_value = tanh(hidden_value);

        output->data[j] = (1 - update_gate) * params->data[j] + update_gate * hidden_value;
        params->data[j] = output->data[j];
    }
    iray1d_free(bias_hidden);
    iray1d_free(bias_update);
    iray1d_free(bias_reset);
}

void layer_gru_backward(Layer *layer, float *delta, float rate) {
    for (size_t j = 0; j < layer->inputSize; j++) {
        for (size_t k = 0; k < layer->outputSize; k++) {
            float dw = rate * delta[k] * layer->input->data[j];
            layer->weight->data[j][k] += dw;
            layer->weight->data[layer->inputSize][k] += dw;
            layer->weight->data[layer->inputSize + 1][k] += dw;
            layer->weight->data[layer->inputSize + 2][k] += dw;
        }
    }

    for (size_t j = 0; j < layer->outputSize; j++) {
        float db = delta[j] * rate;
        layer->bias->data[j] += db;
        layer->bias->data[layer->outputSize + j] += db;
        layer->bias->data[layer->outputSize*2 + j] += db;
    }
}

void layer_activation_sigmoid_forward(Layer *layer) {
    for (size_t i = 0; i < layer->outputSize; i++)
    {
        layer->output->data[i] = sigmoid(layer->input->data[i]);
    }
}
void layer_activation_tanh_forward(Layer *layer) {
    for (size_t i = 0; i < layer->outputSize; i++)
    {
        layer->output->data[i] = tanhf(layer->input->data[i]);
    }
}
void layer_activation_softmax_forward(Layer *layer) {
    float max_val = layer->input->data[0];
    for (size_t i = 1; i < layer->inputSize; i++) {
        if (layer->input->data[i] > max_val) {
            max_val = layer->input->data[i];
        }
    }
    float sum = 0.0;
    for (size_t i = 0; i < layer->inputSize; i++)
    {
        layer->output->data[i] = expf(layer->input->data[i] - max_val);
        sum += layer->output->data[i];
    }
    for (size_t i = 0; i < layer->inputSize; i++)
    {
        layer->output->data[i] /= sum;
    }
    
}
void layer_activation_sigmoid_backward(Layer *layer, float *delta, float rate) {
    (void)delta;
    (void)rate;
    for (size_t i = 0; i < layer->outputSize; i++)
    {
        layer->input->data[i] = dsigmoid(layer->output->data[i]);
    }
}
void layer_activation_tanh_backward(Layer *layer, float *delta, float rate) {
    (void)delta;
    (void)rate;
    for (size_t i = 0; i < layer->outputSize; i++)
    {
        layer->input->data[i] = dtanh(layer->output->data[i]);
    }
}
void layer_activation_softmax_backward(Layer *layer, float *delta, float rate) {
    (void)delta;
    (void)rate;
    for (size_t i = 0; i < layer->inputSize; i++) {
        float softmax_i = layer->output->data[i];
        for (size_t j = 0; j < layer->outputSize; j++) {
            if (i == j) {
                layer->input->data[i] *= softmax_i * (1.0 - softmax_i);
            } else {
                layer->input->data[i] *= -softmax_i * layer->output->data[j];
            }
        }
    }
}

void layer_dropout_forward(Layer *layer) {
    for (size_t i = 0; i < layer->outputSize; i++)
    {
        float rand = random_uniform(0, 1);
        if(rand < layer->params->data[0]) {
            layer->output->data[i] = 0;    
        }else {
            layer->output->data[i] = layer->input->data[i];
        }
    }
}

void layer_dropout_backward(Layer *layer, float *delta, float rate) {
    (void)delta;
    (void)rate;
    for (size_t i = 0; i < layer->inputSize; i++) {
        layer->input->data[i] = layer->output->data[i];
    }
}

void layer_shuffle_forward(Layer *layer) {
    Iray1D *used_indices_e = iray1d_alloc(layer->outputSize);
    int source_index;
    Iray1D *used_indices = iray1d_fill(used_indices_e, 0);
    iray1d_free(used_indices_e);
    for (size_t i = layer->outputSize - 1; i > 0; i--)
    {
        source_index = rand() % layer->outputSize;
        while (used_indices->data[source_index]) {
            source_index = rand() % layer->outputSize;
        }
        used_indices->data[source_index] = 1;
        layer->output->data[i] = layer->input->data[source_index];
    }
    iray1d_free(used_indices);
}

void layer_shuffle_backward(Layer *layer, float *delta, float rate) {
    (void)delta;
    (void)rate;
    for (size_t i = 0; i < layer->inputSize; i++) {
        layer->input->data[i] = layer->output->data[i];
    }
}

Layer *layer_dense(size_t inputSize, size_t outputSize) {
    return layer_alloc(Dense, inputSize, outputSize, 0, layer_dense_forward, layer_dense_backward);
}

Layer *layer_rnn(size_t inputSize, size_t outputSize) {
    Layer *layer = layer_alloc(Dense, inputSize, outputSize, outputSize, layer_rnn_forward, layer_rnn_backward);
    iray2d_free(layer->weight);
    layer->weight = iray2d_alloc(inputSize + 1, outputSize);
    return layer;
}

Layer *layer_gru(size_t inputSize, size_t outputSize) {
    Layer *layer = layer_alloc(Dense, inputSize, outputSize, outputSize, layer_gru_forward, layer_gru_backward);
    iray1d_free(layer->bias);
    iray2d_free(layer->weight);
    layer->bias = iray1d_alloc(outputSize * 3);
    layer->weight = iray2d_alloc(inputSize + 3, outputSize);
    return layer;
}

Layer *layer_activation(ActivationTypes activation) {
    Layer *layer = layer_alloc(Activation, 0, 0, 1, NULL, NULL);
    layer->params->data[0] = activation;
    switch (activation)
    {
    case Sigmoid:
        layer->forward = layer_activation_sigmoid_forward;
        layer->backward = layer_activation_sigmoid_backward;
        break;
    case Tanh:
        layer->forward = layer_activation_tanh_forward;
        layer->backward = layer_activation_tanh_backward;
        break;
    case Softmax:
        layer->forward = layer_activation_softmax_forward;
        layer->backward = layer_activation_softmax_backward;
    }
    return layer;
}

Layer *layer_dropout(float rate) {
    ISERT_MSG(1 > rate, "Rate should be less then 1");
    ISERT_MSG(rate > 0, "Rate should be more then 0");
    Layer *layer = layer_alloc(Dropout, 0, 0, 1, layer_dropout_forward, layer_dropout_backward);
    layer->params->data[0] = rate;
    return layer;
}

Layer *layer_shuffle(float rate) {
    ISERT_MSG(1 > rate, "Rate should be less then 1");
    ISERT_MSG(rate > 0, "Rate should be more then 0");
    Layer *layer = layer_alloc(Shuffle, 0, 0, 1, layer_shuffle_forward, layer_shuffle_backward);
    layer->params->data[0] = rate;
    return layer;
}

#endif // ILAYER_H