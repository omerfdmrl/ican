#ifndef INITIALIZER_H

#define INITIALIZER_H

#include "ican.h"

void initializer_ones(Model *model) {
    for (size_t i = 0; i < model->layer_count; i++)
    {
        for (size_t j = 0; j < model->layers[i]->weight->rows; j++)
        {
            for (size_t k = 0; k < model->layers[i]->weight->cols; k++)
            {
                model->layers[i]->weight->data[j][k] = 1;
            }
        }
        for (size_t j = 0; j < model->layers[i]->bias->rows; j++)
        {
            model->layers[i]->bias->data[j] = 1;
        }
    }
}

void initializer_random_heuniform(Model *model) {
    for (size_t i = 0; i < model->layer_count; i++)
    {
        for (size_t j = 0; j < model->layers[i]->weight->rows; j++)
        {
            for (size_t k = 0; k < model->layers[i]->weight->cols; k++)
            {
                model->layers[i]->weight->data[j][k] = random_heuniform(model->layers[i]->inputSize, model->layers[i]->outputSize);
            }
        }
        for (size_t j = 0; j < model->layers[i]->bias->rows; j++)
        {
            model->layers[i]->bias->data[j] = random_heuniform(model->layers[i]->inputSize, model->layers[i]->outputSize);
        }
    }
}

void initializer_random_normal(Model *model, va_list args) {
    float mean = va_arg(args, double);
    float stddev = va_arg(args, double);
    if(stddev == 0.0) {
        stddev = 0.05;
    }
    ASSERT_MSG(mean != stddev, "Mean(first) can not be equal to Stddev(second)");
    for (size_t i = 0; i < model->layer_count; i++)
    {
        for (size_t j = 0; j < model->layers[i]->weight->rows; j++)
        {
            for (size_t k = 0; k < model->layers[i]->weight->cols; k++)
            {
                model->layers[i]->weight->data[j][k] = random_normal(mean, stddev);
            }
        }
        for (size_t j = 0; j < model->layers[i]->bias->rows; j++)
        {
            model->layers[i]->bias->data[j] = random_normal(mean, stddev);
        }
    }
}

void initializer_random_uniform(Model *model, va_list args) {
    float low = va_arg(args, double);
    float high = va_arg(args, double);
    if(low == 0.0) {
        low = -0.05;
    }
    if(high == 0.0) {
        high = 0.05;
    }
    ASSERT_MSG(high > low, "High variable(first) must be higher than low(second) variable.");
    for (size_t i = 0; i < model->layer_count; i++)
    {
        for (size_t j = 0; j < model->layers[i]->weight->rows; j++)
        {
            for (size_t k = 0; k < model->layers[i]->weight->cols; k++)
            {
                model->layers[i]->weight->data[j][k] = random_uniform(low, high);
            }
        }
        for (size_t j = 0; j < model->layers[i]->bias->rows; j++)
        {
            model->layers[i]->bias->data[j] = random_uniform(low, high);
        }
    }
}

void initializer_random_xavier(Model *model) {
    for (size_t i = 0; i < model->layer_count; i++)
    {
        float a = random_xavier(model->layers[i]->inputSize, model->layers[i]->outputSize);
        for (size_t j = 0; j < model->layers[i]->weight->rows; j++)
        {
            for (size_t k = 0; k < model->layers[i]->weight->cols; k++)
            {
                model->layers[i]->weight->data[j][k] = random_xavier_rand(a);
            }
        }
        for (size_t j = 0; j < model->layers[i]->bias->rows; j++)
        {
            model->layers[i]->bias->data[j] = random_xavier_rand(a);
        }
    }
}

void initializer_zeros(Model *model) {
    for (size_t i = 0; i < model->layer_count; i++)
    {
        for (size_t j = 0; j < model->layers[i]->weight->rows; j++)
        {
            for (size_t k = 0; k < model->layers[i]->weight->cols; k++)
            {
                model->layers[i]->weight->data[j][k] = 0;
            }
        }
        for (size_t j = 0; j < model->layers[i]->bias->rows; j++)
        {
            model->layers[i]->bias->data[j] = 0;
        }
    }
}

#endif // !INITIALIZER_H