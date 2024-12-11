#include "machine_learning/supervised_learning/neural_network.h"

float sigmoid(float x) {
	return 1.f / (1.f + expf(-x));
}
float dsigmoid(float x) {
	return x * (1 - x);
}
float dtanh(float x) {
	return 1.0 / (coshf(x) * coshf(x));
}
float relu(float x) {
    return fmaxf(0.0f, x);
}
float drelu(float x) {
    return fmaxf(0.0f, x);
}

float random_uniform(float low,float high) {
	return ((float) rand() / (float) RAND_MAX) * (high - low) + low;
}
float random_normal(float mean, float stddev) {
    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;

    double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    
    return mean + z0 * stddev;
}
float random_xavier(float fan_in, float fan_out) {
    float a = sqrt(6.0 / (fan_in + fan_out));
    return ((float)rand() / RAND_MAX) * 2.0 * a - a;
}
float random_xavier_sqrt(float fan_in, float fan_out) {
    float a = sqrt(6.0 / (fan_in + fan_out));
    return a;
}
float random_xavier_rand(float a) {
    return ((float)rand() / RAND_MAX) * 2.0 * a - a;
}
float random_heuniform(float fan_in, float fan_out) {
    float limit = sqrt(6.0 / (fan_in + fan_out));
    return ((float)rand() / RAND_MAX) * 2.0 * limit - limit;
}

NNData nndata_alloc(Array1D *array1d, Array2D *array2d) {
    NNData output;
    if (array1d != NULL) {
        output.array1d = array1d;
    } else if (array2d != NULL) {
        output.array2d = array2d;
    } else {
        output.array1d = NULL;
    }
    return output;
}
void nndata_free(NNData data, int shape) {
    if (shape == 1) {
        array1d_free(data.array1d);
    } else if(shape == 2) {
        array2d_free(data.array2d);
    }
}
NNDataSet *nndataset_1d_from(float *data, int64 size, int64 rows) {
    NNDataSet *dataset = (NNDataSet *)ICAN_MALLOC(sizeof(NNDataSet));
    dataset->size = size;
    dataset->shape = array1d;
    dataset->data = (NNData *) ICAN_MALLOC(sizeof(NNData) * size);
    for (size_t i = 0; i < size; i++) {
        dataset->data[i].array1d = array1d_from(data+(i * rows), rows);
    }
    return dataset;
}
void nndataset_free(NNDataSet *dataset) {
    for (size_t i = 0; i < dataset->size; i++) {
        nndata_free(dataset->data[i], (int) dataset->shape);
    }
    ICAN_FREE(dataset->data);
    ICAN_FREE(dataset);
    
}

Layer *layer_alloc(LayerName name, LayerShape shape, int64 input_size, int64 output_size, int64 param_size, NNData (*forward)(Layer *layer, NNData input), NNData (*backward)(Layer *layer, NNData input, Array1D *delta, float rate)) {
    Layer *layer = (Layer *)ICAN_MALLOC(sizeof(Layer));
    layer->name = name;
    layer->shape = shape;
    layer->input_size = input_size;
    layer->output_size = output_size;
    layer->param = array1d_alloc(param_size);
    layer->weight = array2d_alloc(input_size, output_size);
    layer->bias = array1d_alloc(output_size);
    layer->forward = forward;
    layer->backward = backward;
    return layer;
}

void layer_free(Layer *layer) {
    array1d_free(layer->param);
    array1d_free(layer->bias);
    array2d_free(layer->weight);
    ICAN_FREE(layer);
}

void layer_print(Layer *layer) {
    int64 param = layer->input_size * layer->output_size + layer->bias->rows;
	printf("%-20s(%llu, %llu)%-16s%llu\n", LAYER_NAME(layer->name), layer->input_size, layer->output_size,"", param);
}

NNData layer_dense_forward(Layer *layer, NNData input) {
    Array1D *output = array2d_dot_array1d_sum(layer->weight, input.array1d);
    array1d_add_inplace(output, layer->bias);
    return nndata_alloc(output, NULL);
}

NNData layer_dense_backward(Layer *layer, NNData input, Array1D *delta, float rate) {
    for (size_t j = 0; j < layer->input_size; j++) {
        for (size_t k = 0; k < layer->output_size; k++) {
            float dw = rate * delta->data[k] * input.array1d->data[j];
            layer->weight->data[j * layer->weight->cols + k] += dw;
        }
    }

    for (size_t j = 0; j < layer->output_size; j++) {
        float db = delta->data[j] * rate;
        layer->bias->data[j] += db;
    }
    return nndata_alloc(NULL, NULL);
}

Layer *layer_dense(int64 input_size, int64 output_size) {
    return layer_alloc(Dense, array1d, input_size, output_size, 0, layer_dense_forward, layer_dense_backward);
}

NNData layer_activation_sigmoid_forward(Layer *layer, NNData input) {
    return nndata_alloc(array1d_apply(input.array1d, sigmoid), NULL);
}
NNData layer_activation_tanh_forward(Layer *layer, NNData input) {
    return nndata_alloc(array1d_apply(input.array1d, tanhf), NULL);
}
NNData layer_activation_softmax_forward(Layer *layer, NNData input) {
    float max_val = array1d_max(input.array1d);
    Array1D *dotted = array1d_sub_scalar(input.array1d, max_val);
    array1d_apply_inplace(dotted, expf);
    float sum = array1d_sum(dotted);
    return nndata_alloc(array1d_sub_scalar(dotted, sum), NULL);
}
NNData layer_activation_relu_forward(Layer *layer, NNData input) {
    return nndata_alloc(array1d_apply(input.array1d, relu), NULL);
}
NNData layer_activation_sigmoid_backward(Layer *layer, NNData input, Array1D *delta, float rate) {
    (void)delta;
    (void)rate;
    return nndata_alloc(array1d_apply(input.array1d, dsigmoid), NULL);
}
NNData layer_activation_tanh_backward(Layer *layer, NNData input, Array1D *delta, float rate) {
    (void)delta;
    (void)rate;
    return nndata_alloc(array1d_apply(input.array1d, dtanh), NULL);
}
NNData layer_activation_softmax_backward(Layer *layer, NNData input, Array1D *delta, float rate) {
    (void)delta;
    (void)rate;
    Array1D *output = array1d_alloc(layer->output_size);
    for (size_t i = 0; i < layer->input_size; i++) {
        float softmax_i = input.array1d->data[i];
        for (size_t j = 0; j < layer->output_size; j++) {
            if (i == j) {
                output->data[i] *= softmax_i * (1.0 - softmax_i);
            } else {
                output->data[i] *= -softmax_i * input.array1d->data[j];
            }
        }
    }
    return nndata_alloc(output, NULL);
}
NNData layer_activation_relu_backward(Layer *layer, NNData input, Array1D *delta, float rate) {
    (void)delta;
    (void)rate;
    return nndata_alloc(array1d_apply(input.array1d, drelu), NULL);
}

Layer *layer_activation(ActivationType activation) {
    Layer *layer = layer_alloc(Activation, array1d, 0, 0, 1, NULL, NULL);
    layer->param->data[0] = activation;
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
    case Relu:
        layer->forward = layer_activation_relu_forward;
        layer->backward = layer_activation_relu_backward;
    }
    return layer;
}

NNData layer_dropout_forward(Layer *layer, NNData input) {
    Array1D *output = array1d_alloc(layer->output_size);
    for (size_t i = 0; i < layer->output_size; i++)
    {
        float rand = random_uniform(0, 1);
        if(rand < layer->param->data[0]) {
            output->data[i] = 0;    
        }else {
            output->data[i] = input.array1d->data[i];
        }
    }
    return nndata_alloc(output, NULL);
}

NNData layer_dropout_backward(Layer *layer, NNData input, Array1D *delta, float rate) {
    (void)delta;
    (void)rate;
    return input;
}

Layer *layer_dropout(float rate) {
    ASSERT_MSG(1 > rate, "Rate should be less then 1");
    ASSERT_MSG(rate > 0, "Rate should be more then 0");
    Layer *layer = layer_alloc(Dropout, array1d, 0, 0, 1, layer_dropout_forward, layer_dropout_backward);
    layer->param->data[0] = rate;
    return layer;
}

NNData layer_shuffle_forward(Layer *layer, NNData input) {
    Array1D *output = array1d_alloc(layer->output_size);
    Array1D *used_indices = array1d_alloc(layer->output_size);
    int source_index;
    array1d_fill_inplace(used_indices, 0);
    for (size_t i = layer->output_size - 1; i > 0; i--) {
        source_index = rand() % layer->output_size;
        while (used_indices->data[source_index]) {
            source_index = rand() % layer->output_size;
        }
        used_indices->data[source_index] = 1;
        output->data[i] = input.array1d->data[source_index];
    }
    array1d_free(used_indices);
    return nndata_alloc(output, NULL);
}

NNData layer_shuffle_backward(Layer *layer, NNData input, Array1D *delta, float rate) {
    (void)delta;
    (void)rate;
    return input;
}

Layer *layer_shuffle(float rate) {
    ASSERT_MSG(1 > rate, "Rate should be less then 1");
    ASSERT_MSG(rate > 0, "Rate should be more then 0");
    Layer *layer = layer_alloc(Shuffle, array1d, 0, 0, 1, layer_shuffle_forward, layer_shuffle_backward);
    layer->param->data[0] = rate;
    return layer;
}

void compiler_update(Compiler *compiler, int t) {
    if (compiler->schedule == WarmupDecay) {
        compiler->lr = powf(compiler->d, -0.5) * fminf(powf(t, -0.5), t * powf(compiler->warmup_steps, -1.5));
    } else if (compiler->schedule == CosineAnnealing) {
        compiler->lr = compiler->lr_min + 
            0.5 * (compiler->lr_max - compiler->lr_min) * (1 + cosf((float)t / compiler->e * M_PI));
    }
}

Compiler *compiler_alloc(Optimizer optimizer, CompilerSchedule schedule) {
    Compiler *c = (Compiler *) ICAN_MALLOC(sizeof(Compiler));
    c->optimizer = optimizer;
    c->schedule = schedule;
    c->eps = 1e-5;
    c->lr = 1e-5;
    c->lr_min = 1e-5;
    c->lr_max = c->lr;
    c->warmup_steps = 4000;
    c->d = 512;
    c->e = 0;
    c->update = compiler_update;
    return c;
};

void compiler_free(Compiler *compiler) {
    ICAN_FREE(compiler);
}

Model *model_alloc(int16 layer_count) {
    Model *m = (Model *)ICAN_MALLOC(sizeof(Model));
    m->layer_count = 0;
    m->layers = (Layer **)ICAN_MALLOC(sizeof(Layer) * layer_count);
    m->compiler = (Compiler *) NULL;
    return m;
}

void model_free(Model *model) {
    for (size_t i = 0; i < model->layer_count; i++) {
        layer_free(model->layers[i]);
    }
    ICAN_FREE(model->layers);
    compiler_free(model->compiler);
    ICAN_FREE(model);
}

void model_add(Model *model, Layer *layer) {
	if(model->layer_count != 0) {
		if(layer->input_size == 0) {
			layer->input_size = model->layers[model->layer_count - 1]->output_size;
		}
		if(layer->output_size == 0) {
			layer->output_size = layer->input_size;
		}
	}
	model->layers[model->layer_count] = layer;
	model->layer_count++;
}

NNData model_forward(Model *model, NNData input) {
    NNData layer_input = model->layers[0]->forward(model->layers[0], input);
    NNData temp_input;
	for (size_t i = 1; i < model->layer_count; i++) {
		temp_input = model->layers[i]->forward(model->layers[i], layer_input);
        nndata_free(layer_input, (int) model->layers[i]->shape);
        layer_input = temp_input;
    }
    return layer_input;
}

float model_cost(Model *model, NNDataSet *inputs, NNDataSet *outputs) {
    float c = 0;
    float d = 0;
    for (size_t i = 0; i < inputs->size; i++) {
        NNData input = inputs->data[i];
        NNData output = outputs->data[i];
        NNData predict = model_forward(model, input);
        LayerShape s = inputs->shape;
        if (s == array1d) {
            for (size_t i = 0; i < output.array1d->rows; i++) {
                d = output.array1d->data[i] - predict.array1d->data[i];
                c += d*d;
            }
            c /= output.array1d->rows;
        }else if (s == array2d) {
            for (size_t i = 0; i < output.array2d->rows; i++) {
                for (size_t j = 0; j < output.array2d->cols; j++) {
                    d = output.array2d->data[i * output.array2d->cols + j] - predict.array2d->data[i * output.array2d->cols + j];
                    c += d*d;
                }
            }
            c /= output.array2d->rows * output.array2d->cols;
        }
    }
    return c;
}

void randomizer_ones(Model *model) {
    for (size_t i = 0; i < model->layer_count; i++) {
        for (size_t j = 0; j < model->layers[i]->weight->rows; j++) {
            for (size_t k = 0; k < model->layers[i]->weight->cols; k++) {
                model->layers[i]->weight->data[j * model->layers[i]->weight->cols + k] = 1;
            }
        }
        for (size_t j = 0; j < model->layers[i]->bias->rows; j++) {
            model->layers[i]->bias->data[j] = 1;
        }
    }
}

void randomizer_random_heuniform(Model *model) {
    for (size_t i = 0; i < model->layer_count; i++) {
        for (size_t j = 0; j < model->layers[i]->weight->rows; j++) {
            for (size_t k = 0; k < model->layers[i]->weight->cols; k++) {
                model->layers[i]->weight->data[j * model->layers[i]->weight->cols + k] = random_heuniform(model->layers[i]->input_size, model->layers[i]->output_size);
            }
        }
        for (size_t j = 0; j < model->layers[i]->bias->rows; j++) {
            model->layers[i]->bias->data[j] = random_heuniform(model->layers[i]->input_size, model->layers[i]->output_size);
        }
    }
}

void randomizer_random_normal(Model *model, va_list args) {
    float mean = va_arg(args, double);
    float stddev = va_arg(args, double);
    if(stddev == 0.0) {
        stddev = 0.05;
    }
    ASSERT_MSG(mean != stddev, "Mean(first) can not be equal to Stddev(second)");
    for (size_t i = 0; i < model->layer_count; i++) {
        for (size_t j = 0; j < model->layers[i]->weight->rows; j++) {
            for (size_t k = 0; k < model->layers[i]->weight->cols; k++) {
                model->layers[i]->weight->data[j * model->layers[i]->weight->cols + k] = random_normal(mean, stddev);
            }
        }
        for (size_t j = 0; j < model->layers[i]->bias->rows; j++) {
            model->layers[i]->bias->data[j] = random_normal(mean, stddev);
        }
    }
}

void randomizer_random_uniform(Model *model, va_list args) {
    float low = va_arg(args, double);
    float high = va_arg(args, double);
    if(low == 0.0) {
        low = -0.05;
    }
    if(high == 0.0) {
        high = 0.05;
    }
    ASSERT_MSG(high > low, "High variable(first) must be higher than low(second) variable.");
    for (size_t i = 0; i < model->layer_count; i++) {
        for (size_t j = 0; j < model->layers[i]->weight->rows; j++) {
            for (size_t k = 0; k < model->layers[i]->weight->cols; k++) {
                model->layers[i]->weight->data[j * model->layers[i]->weight->cols + k] = random_uniform(low, high);
            }
        }
        for (size_t j = 0; j < model->layers[i]->bias->rows; j++) {
            model->layers[i]->bias->data[j] = random_uniform(low, high);
        }
    }
}

void randomizer_random_xavier(Model *model) {
    for (size_t i = 0; i < model->layer_count; i++) {
        float a = random_xavier(model->layers[i]->input_size, model->layers[i]->output_size);
        for (size_t j = 0; j < model->layers[i]->weight->rows; j++) {
            for (size_t k = 0; k < model->layers[i]->weight->cols; k++) {
                model->layers[i]->weight->data[j * model->layers[i]->weight->cols + k] = random_xavier_rand(a);
            }
        }
        for (size_t j = 0; j < model->layers[i]->bias->rows; j++) {
            model->layers[i]->bias->data[j] = random_xavier_rand(a);
        }
    }
}

void randomizer_zeros(Model *model) {
    for (size_t i = 0; i < model->layer_count; i++) {
        for (size_t j = 0; j < model->layers[i]->weight->rows; j++) {
            for (size_t k = 0; k < model->layers[i]->weight->cols; k++) {
                model->layers[i]->weight->data[j * model->layers[i]->weight->cols + k] = 0;
            }
        }
        for (size_t j = 0; j < model->layers[i]->bias->rows; j++) {
            model->layers[i]->bias->data[j] = 0;
        }
    }
}

void model_randomize(Model *model, Randomizer randomizer, ...) {
    void (*randomizerFunction)(Model *model) = NULL;
	void (*randomizerFunctionwa)(Model *model, va_list args) = NULL;;
	switch (randomizer) {
	case Zeros:
		randomizerFunction = randomizer_zeros;
		break;

	case Ones:
		randomizerFunction = randomizer_ones;
		break;
	
	case RandomUniform:
		randomizerFunctionwa = randomizer_random_uniform;
		break;

	case RandomNormal:
		randomizerFunctionwa = randomizer_random_normal;
		break;

	case RandomXavier:
		randomizerFunction = randomizer_random_xavier;
		break;

	case RandomHeUniform:
		randomizerFunction = randomizer_random_heuniform;
		break;
	}
	va_list args;
	va_start(args, randomizer);
	if(randomizerFunction) {
		randomizerFunction(model);
	}else {
		randomizerFunctionwa(model, args);
	}
	va_end(args);
}

void model_print(Model *model) {
    printf("%-20s%-20s%-20s\n", "Layer", "Shape", "Param");
	printf("=================================================\n");
	for (size_t i = 0; i < model->layer_count; i++) {
		layer_print(model->layers[i]);
	}
	printf("=================================================\n");
}

void model_compile(Model *model, Compiler *compiler) {
    int32 d = 0;
    for (size_t i = 0; i < model->layer_count; i++) {
        switch (model->layers[i]->name) {
            case Dense:
                d += model->layers[i]->input_size * model->layers[i]->output_size + model->layers[i]->output_size;
                break;

            case Activation:
            case Dropout:
            case Shuffle:
            case Flatten:
            case Input:
            case Output:
                d += 0;
                break;

            case BatchNormalization:
                d += 2 * model->layers[i]->output_size;
                break;
        }
    }
    compiler->d = d;
    model->compiler = compiler;
}

void optimizer_finite_diff(Model *model, NNDataSet *inputs, NNDataSet *outputs) {
    float cost = model_cost(model, inputs, outputs);
    float eps = model->compiler->eps;
    float gradient;

    for (size_t i = 0; i < model->layer_count; i++) {
        Layer *layer = model->layers[i];
        if (layer->name != Dense) continue;
        for (size_t j = 0; j < layer->input_size; j++) {
            for (size_t k = 0; k < layer->output_size; k++) {
                layer->weight->data[j * layer->weight->cols + k] += eps;
                float new_cost = model_cost(model, inputs, outputs);
                gradient = (new_cost - cost) / eps;
                layer->weight->data[j * layer->weight->cols + k] -= eps;
                layer->weight->data[j * layer->weight->cols + k] -= model->compiler->lr * gradient;
            }
        }

        for (size_t k = 0; k < layer->output_size; k++) {
            layer->bias->data[k] += eps;
            float new_cost = model_cost(model, inputs, outputs);
            gradient = (new_cost - cost) / eps;
            layer->bias->data[k] -= eps;
            layer->bias->data[k] -= model->compiler->lr * gradient;
        }
    }
}

void optimizer_batch_gradient_descent(Model *model, NNDataSet *inputs, NNDataSet *outputs) {
    for (size_t i = 0; i < inputs->size; i++) {
        NNData input = inputs->data[i];
        NNData output = outputs->data[i];
        NNData temp;
        NNData out = nndata_alloc(outputs->data[i].array1d, NULL);

        NNData *nn_inputs = (NNData *)ICAN_MALLOC(sizeof(NNData) * model->layer_count);
        NNData parsed = model->layers[0]->forward(model->layers[0], input);
        nn_inputs[0] = input;
        for (size_t l = 1; l < model->layer_count; l++) {
            nn_inputs[l] = model->layers[l]->forward(model->layers[l], parsed);
            parsed = nn_inputs[l];
        }

        int first = 1;
        Array1D **delta = (Array1D **)ICAN_MALLOC(sizeof(Array1D *) * model->layer_count);
        for (size_t l = model->layer_count; l-- > 0; ) {
            Layer *layer = model->layers[l];
            delta[l] = array1d_alloc(layer->output_size);
            if (layer->weight->rows == 0) {
                temp = layer->backward(layer, nn_inputs[l], delta[l], model->compiler->lr);
                nndata_free(out, 1);
                out = temp;
                continue;
            }
            Layer *prev_layer;
            Array1D *prev_delta;
            for (size_t ll = l + 1; ll < model->layer_count - 1; ll++) {
                if(model->layers[ll]->weight->rows != 0){
                    prev_layer = model->layers[ll];
                    prev_delta = delta[ll];
                    break;
                }
            }
            if (first) {
                for (size_t j = 0; j < output.array1d->rows; j++) {
                    delta[l]->data[j] = (output.array1d->data[j] - parsed.array1d->data[j]) * output.array1d->data[j];
                }
                first = 0;
            }else {
                for (size_t j = 0; j < layer->output_size; j++)
                {
                    delta[l]->data[j] = 0.0;
                    for (size_t k = 0; k < prev_layer->output_size; k++)  
                    {
                        delta[l]->data[j] += prev_delta->data[k] * prev_layer->weight->data[j * prev_layer->weight->cols + k];
                    }
                    delta[l]->data[j] *= output.array1d->data[j];
                }
            }
            temp = layer->backward(layer, nn_inputs[l], delta[l], model->compiler->lr);
            nndata_free(out, 1);
            out = temp;
        }
        
    }
}

void model_fit(Model *model, int32 epochs, NNDataSet *inputs, NNDataSet *outputs) {
    model->compiler->e = epochs;
    for (size_t i = 0; i < epochs; i++) {
        float cost = model_cost(model, inputs, outputs);
        printf("epoch[%zu]: %f\n", i, cost);
        optimizer_batch_gradient_descent(model, inputs, outputs);
    }
}