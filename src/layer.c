#ifndef LAYER_H

#define LAYER_H

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

    Iray2D *weight_reset = iray2d_slice(layer->weight, 0, layer->inputSize);
    Iray2D *weight_update = iray2d_slice(layer->weight, layer->inputSize, layer->inputSize * 2);
    Iray2D *weight_hidden = iray2d_slice(layer->weight, layer->inputSize * 2, layer->inputSize * 3);

    Iray2D *_weight_reset_hidden = iray2d_slice(layer->weight, layer->inputSize * 3, layer->inputSize * 3 + 1);
    Iray2D *_weight_update_hidden = iray2d_slice(layer->weight, layer->inputSize * 3 + 1, layer->inputSize * 3 + 2);
    Iray2D *_weight_hidden_hidden = iray2d_slice(layer->weight, layer->inputSize * 3 + 2, layer->inputSize * 3 + 3);

    Iray1D *weight_reset_hidden = iray1d_alloc(layer->outputSize);
    Iray1D *weight_update_hidden = iray1d_alloc(layer->outputSize);
    Iray1D *weight_hidden_hidden = iray1d_alloc(layer->outputSize);

    for (size_t i = 0; i < layer->outputSize; i++) {
        weight_reset_hidden->data[i] = _weight_reset_hidden->data[0][i];
        weight_update_hidden->data[i] = _weight_update_hidden->data[0][i];
        weight_hidden_hidden->data[i] = _weight_hidden_hidden->data[0][i];
    }
    iray2d_free(_weight_reset_hidden);
    iray2d_free(_weight_update_hidden);
    iray2d_free(_weight_hidden_hidden);

    size_t inputSize = layer->inputSize;
    size_t outputSize = layer->outputSize;

    for (size_t j = 0; j < outputSize; j++) {
        float reset_gate = 0, update_gate = 0, hidden_value = 0;

        for (size_t i = 0; i < inputSize; i++) {
            reset_gate += input->data[i] * weight_reset->data[i][j];
            update_gate += input->data[i] * weight_update->data[i][j];
        }

        reset_gate += params->data[j] * weight_reset_hidden->data[j];
        update_gate += params->data[j] * weight_update_hidden->data[j];
        reset_gate += layer->bias->data[j];
        update_gate += layer->bias->data[outputSize + j];

        reset_gate = sigmoid(reset_gate);
        update_gate = sigmoid(update_gate);

        for (size_t i = 0; i < inputSize; i++) {
            hidden_value += input->data[i] * weight_hidden->data[i][j];
        }
        hidden_value += reset_gate * params->data[j] * weight_hidden_hidden->data[j];
        hidden_value += layer->bias->data[outputSize * 2 + j];
        hidden_value = tanh(hidden_value);

        output->data[j] = (1 - update_gate) * params->data[j] + update_gate * hidden_value;
        params->data[j] = output->data[j];
    }
    iray1d_free(bias_hidden);
    iray1d_free(bias_update);
    iray1d_free(bias_reset);

    iray1d_free(weight_reset_hidden);
    iray1d_free(weight_update_hidden);
    iray1d_free(weight_hidden_hidden);
}

void layer_gru_backward(Layer *layer, float *delta, float rate) {
    for (size_t j = 0; j < layer->inputSize; j++) {
        for (size_t k = 0; k < layer->outputSize; k++) {
            layer->weight->data[j][k] += rate * delta[k] * layer->input->data[j];
            layer->weight->data[layer->inputSize + j][k] += rate * delta[k] * layer->input->data[j];
            layer->weight->data[layer->inputSize * 2 + j][k] += rate * delta[k] * layer->input->data[j];
        }
    }

    for (size_t j = 0; j < layer->outputSize; j++) {
        layer->bias->data[j] += rate * delta[j];
        layer->bias->data[layer->outputSize + j] += rate * delta[j];
        layer->bias->data[layer->outputSize * 2 + j] += rate * delta[j];
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
void layer_activation_relu_forward(Layer *layer) {
    for (size_t i = 0; i < layer->outputSize; i++) {
        layer->output->data[i] = relu(layer->input->data[i]);
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
void layer_activation_relu_backward(Layer *layer, float *delta, float rate) {
    (void)delta;
    (void)rate;
    for (size_t i = 0; i < layer->outputSize; i++) {
        layer->input->data[i] = layer->output->data[i] > 0 ? layer->output->data[i] : 0;
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

void layer_batch_normalization_forward(Layer *layer) {
    float mean = 0.0f;
    float variance = 0.0f;

    for (size_t i = 0; i < layer->inputSize; i++) {
        mean += layer->input->data[i];
    }
    mean /= layer->inputSize;

    for (size_t i = 0; i < layer->inputSize; i++) {
        variance += (layer->input->data[i] - mean) * (layer->input->data[i] - mean);
    }
    variance /= layer->inputSize;

    float epsilon = 1e-5;
    for (size_t i = 0; i < layer->inputSize; i++) {
        float normalized = (layer->input->data[i] - mean) / sqrt(variance + epsilon);
        layer->output->data[i] = normalized * layer->params->data[i] + layer->params->data[layer->inputSize + i];
    }
}

void layer_batch_normalization_backward(Layer *layer, float *delta, float rate) {
    Iray1D *input = layer->input;
    Iray1D *output = layer->output;
    Iray1D *gamma = iray1d_slice(layer->params, 0, layer->inputSize);
    Iray1D *beta = iray1d_slice(layer->params, layer->inputSize, layer->inputSize * 2);
    size_t outputSize = layer->outputSize;

    for (size_t j = 0; j < outputSize; j++) {
        float db = delta[j];
        beta->data[j] += rate * db;

        float dgamma = delta[j] * output->data[j];
        gamma->data[j] += rate * dgamma;
    }

    for (size_t i = 0; i < input->rows; i++) {
        float dinput = 0;

        for (size_t j = 0; j < outputSize; j++) {
            dinput += delta[j] * gamma->data[j] / outputSize;
        }

        input->data[i] += rate * dinput;
    }

    free(beta);
}

Layer *layer_dense(size_t inputSize, size_t outputSize) {
    return layer_alloc(Dense, inputSize, outputSize, 0, layer_dense_forward, layer_dense_backward);
}

Layer *layer_rnn(size_t inputSize, size_t outputSize) {
    Layer *layer = layer_alloc(RNN, inputSize, outputSize, outputSize, layer_rnn_forward, layer_rnn_backward);
    iray2d_free(layer->weight);
    layer->weight = iray2d_alloc(inputSize + 1, outputSize);
    return layer;
}

Layer *layer_gru(size_t inputSize, size_t outputSize) {
    Layer *layer = layer_alloc(GRU, inputSize, outputSize, outputSize, layer_gru_forward, layer_gru_backward);
    iray1d_free(layer->bias);
    iray2d_free(layer->weight);
    layer->bias = iray1d_alloc(outputSize * 3);
    layer->weight = iray2d_alloc(inputSize * 3 + 3, outputSize);
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
    case RELU:
        layer->forward = layer_activation_relu_forward;
        layer->backward = layer_activation_relu_backward;
    }
    return layer;
}

Layer *layer_dropout(float rate) {
    ASSERT_MSG(1 > rate, "Rate should be less then 1");
    ASSERT_MSG(rate > 0, "Rate should be more then 0");
    Layer *layer = layer_alloc(Dropout, 0, 0, 1, layer_dropout_forward, layer_dropout_backward);
    layer->params->data[0] = rate;
    return layer;
}

Layer *layer_shuffle(float rate) {
    ASSERT_MSG(1 > rate, "Rate should be less then 1");
    ASSERT_MSG(rate > 0, "Rate should be more then 0");
    Layer *layer = layer_alloc(Shuffle, 0, 0, 1, layer_shuffle_forward, layer_shuffle_backward);
    layer->params->data[0] = rate;
    return layer;
}

Layer *layer_batch_normalization(size_t inputSize) {
    Layer *layer = layer_alloc(BatchNormalization, inputSize, inputSize, inputSize * 2, layer_batch_normalization_forward, layer_batch_normalization_backward);
    for (size_t i = 0; i < inputSize; i++) {
        layer->params->data[i] = 1.0f;
        layer->params->data[inputSize + i] = 0.0f;
    }
    
    return layer;
}

#endif // LAYER_H