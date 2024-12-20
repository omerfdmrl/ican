#ifndef MODEL_H
#define MODEL_H

#include "ican.h"

Model *model_alloc(size_t layerCount) {
	Model *m = ICAN_MALLOC(sizeof(Model));
	m->layer_count = 0;
	m->layers = ICAN_MALLOC(sizeof(Layer *) * layerCount);
	m->step = NULL;
	return m;
}

void model_add(Model *model, Layer *layer) {
	if(model->layer_count != 0) {
		if(layer->inputSize == 0) {
			layer->inputSize = model->layers[model->layer_count - 1]->outputSize;
			iray1d_free(layer->input);
			layer->input = iray1d_alloc(layer->inputSize);
		}
		if(layer->outputSize == 0) {
			layer->outputSize = layer->inputSize;
			iray1d_free(layer->output);
			layer->output = iray1d_alloc(layer->outputSize);
		}
	}

	model->layers[model->layer_count] = layer;
	model->layer_count++;
}
void model_randomize(Initializers initializer, Model *model, ...) {
	void (*initializerFunction)(Model *model) = NULL;
	void (*initializerFunctionwa)(Model *model, va_list args) = NULL;;
	switch (initializer)
	{
	case Zeros:
		initializerFunction = initializer_zeros;
		break;

	case Ones:
		initializerFunction = initializer_ones;
		break;
	
	case RandomUniform:
		initializerFunctionwa = initializer_random_uniform;
		break;

	case RandomNormal:
		initializerFunctionwa = initializer_random_normal;
		break;

	case RandomXavier:
		initializerFunction = initializer_random_xavier;
		break;

	case RandomHeUniform:
		initializerFunction = initializer_random_heuniform;
		break;
	}
	va_list args;
	va_start(args, model);
	if(initializerFunction) {
		initializerFunction(model);
	}else {
		initializerFunctionwa(model, args);
	}
	va_end(args);
}
void model_forward(Model *model) {
	model->layers[0]->forward(model->layers[0]);
	for (size_t i = 1; i < model->layer_count; i++)
	{
		for (size_t j = 0; j < model->layers[i]->inputSize; j++) {
			model->layers[i]->input->data[j] = model->layers[i - 1]->output->data[j];
		}
		model->layers[i]->forward(model->layers[i]);
	}
}
void model_input(Model *model, float *input) {
	for (size_t i = 0; i < model->layers[0]->inputSize; i++)
	{
		model->layers[0]->input->data[i] = input[i];
	}
}
float model_cost(Model *model, float *input, float *output) {
	model_input(model, input);
	model_forward(model);
	float c = 0;
	float d = 0;
	size_t n = model->layers[model->layer_count - 1]->outputSize;
	for (size_t i = 0; i < n; i++)
	{
		d = (output[i] - model->layers[model->layer_count - 1]->output->data[i]);
		c += d * d;
	}
	return c / n;
}
Iray1D *model_learn(Itimizers optimizer, Model *model, size_t epoch, Iray2D *inputs, Iray2D *outputs, ...) {
	ASSERT_MSG(inputs->cols == model->layers[0]->inputSize, "Input Size should be equal to first layers");
	ASSERT_MSG(outputs->cols == model->layers[model->layer_count - 1]->outputSize, "Output Size should be equal to last layer");
	void (*optimizerFunction)(Model *model, Iray2D *inputs, Iray2D *outputs, va_list args);
	va_list args;
	va_list cpyargs;
	va_start(args, outputs);
	switch (optimizer)
	{
		case FiniteDiff:
			optimizerFunction = optimizer_finite_diff;
			break;

		case BatchGradientDescent:
			optimizerFunction = optimizer_batch_gradient_descent;
			break;
	}
	clock_t timer = clock();
	Iray1D *loss_history = iray1d_alloc(epoch);
	float cost;
	for (size_t e = 0; e < epoch; e++)
	{
		cost = model_cost(model, inputs->data[0], outputs->data[0]);
        print_progress_header(e,epoch, cost);
		va_copy(cpyargs, args);
		optimizerFunction(model, inputs, outputs, cpyargs);
		va_end(cpyargs);
		loss_history->data[e] = cost;
		if(model->step != NULL) model->step(loss_history);
	}
	timer = clock() - timer;
	print_progress_footer(timer, model_cost(model, inputs->data[0], outputs->data[0]));
	va_end(args);
	printf("\n");
	return loss_history;
}
void model_print(Model *model) {
	printf("%-20s%-20s%-20s\n", "Layer", "Shape", "Param");
	printf("=================================================\n");
	for (size_t i = 0; i < model->layer_count; i++)
	{
		layer_print(model->layers[i]);
	}
	printf("=================================================\n");
}
void model_free(Model *model) {
	for (size_t i = 0; i < model->layer_count; i++)
	{
		layer_free(model->layers[i]);
	}
	ICAN_FREE(model->layers);
	ICAN_FREE(model);
}

#endif // !MODEL_H