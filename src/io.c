#ifndef IO_H

#define IO_H

#include "ican.h"

void io_export(Model *model, const char *fileName) {
    cJSON *json_model = cJSON_CreateObject();
    cJSON_AddNumberToObject(json_model, "Layer Count", model->layer_count);

    cJSON *layers = cJSON_CreateArray();
    for (int i = 0; i < model->layer_count; i++) {
        Layer *layer = model->layers[i];
        cJSON *json_layer = cJSON_CreateObject();

        cJSON_AddStringToObject(json_layer, "Type", LAYER_NAME(layer->name));
        cJSON_AddNumberToObject(json_layer, "Inputs", layer->inputSize);
        cJSON_AddNumberToObject(json_layer, "Outputs", layer->outputSize);

        if (layer->bias) {
            cJSON *biases = cJSON_CreateArray();
            for (size_t j = 0; j < layer->bias->rows; j++) {
                cJSON_AddItemToArray(biases, cJSON_CreateNumber(layer->bias->data[j]));
            }
            cJSON_AddItemToObject(json_layer, "Biases", biases);
        }

        if (layer->weight) {
            cJSON *weights = cJSON_CreateArray();
            for (size_t j = 0; j < layer->weight->rows; j++) {
                cJSON *weight_row = cJSON_CreateArray();
                for (size_t k = 0; k < layer->weight->cols; k++) {
                    cJSON_AddItemToArray(weight_row, cJSON_CreateNumber(layer->weight->data[j][k]));
                }
                cJSON_AddItemToArray(weights, weight_row);
            }
            cJSON_AddItemToObject(json_layer, "Weights", weights);
        }

        if (layer->params) {
            cJSON *params = cJSON_CreateArray();
            for (size_t j = 0; j < layer->params->rows; j++) {
                cJSON_AddItemToArray(params, cJSON_CreateNumber(layer->params->data[j]));
            }
            cJSON_AddItemToObject(json_layer, "Params", params);
        }

        cJSON_AddItemToArray(layers, json_layer);
    }
    cJSON_AddItemToObject(json_model, "Layers", layers);

    FILE *fp = fopen(fileName, "w");
    if (fp) {
        char *json_string = cJSON_Print(json_model);
        fprintf(fp, "%s", json_string);
        fclose(fp);
        ICAN_FREE(json_string);
    }

    cJSON_Delete(json_model);
}

Model *io_import(const char *fileName) {
    FILE *fp = fopen(fileName, "r");
    if(fp != NULL, "File could not be opened");

    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    
    char *buffer = (char *)ICAN_MALLOC(file_size + 1);
    fread(buffer, 1, file_size, fp);
    buffer[file_size] = '\0';
    fclose(fp);

    cJSON *json_model = cJSON_Parse(buffer);
    ICAN_FREE(buffer);
    if (!json_model) {
        printf("Error parsing JSON\n");
        return NULL;
    }

    Model *model = model_alloc(cJSON_GetObjectItem(json_model, "Layer Count")->valueint);
    cJSON *layers = cJSON_GetObjectItem(json_model, "Layers");
    cJSON *layer = NULL;
    cJSON_ArrayForEach(layer, layers) {
        const char *type = cJSON_GetObjectItem(layer, "Type")->valuestring;
        size_t inputs = cJSON_GetObjectItem(layer, "Inputs")->valueint;
        size_t outputs = cJSON_GetObjectItem(layer, "Outputs")->valueint;

        Layer *new_layer = NULL;
        if (strcmp(type, "Dense") == 0) {
            new_layer = layer_dense(inputs, outputs);
        } else if (strcmp(type, "Activation") == 0) {
            cJSON *params = cJSON_GetObjectItem(layer, "Params");
            ActivationTypes activationType = (ActivationTypes)cJSON_GetArrayItem(params, 0)->valueint;
            new_layer = layer_activation(activationType);
        }

        cJSON *biases = cJSON_GetObjectItem(layer, "Biases");
        if (biases) {
            for (size_t i = 0; i < cJSON_GetArraySize(biases); i++) {
                new_layer->bias->data[i] = cJSON_GetArrayItem(biases, i)->valuedouble;
            }
        }

        cJSON *weights = cJSON_GetObjectItem(layer, "Weights");
        if (weights) {
            for (size_t i = 0; i < cJSON_GetArraySize(weights); i++) {
                cJSON *weight_row = cJSON_GetArrayItem(weights, i);
                for (size_t j = 0; j < cJSON_GetArraySize(weight_row); j++) {
                    new_layer->weight->data[i][j] = cJSON_GetArrayItem(weight_row, j)->valuedouble;
                }
            }
        }

        cJSON *params = cJSON_GetObjectItem(layer, "Params");
        if (params) {
            for (size_t i = 0; i < cJSON_GetArraySize(params); i++) {
                new_layer->params->data[i] = cJSON_GetArrayItem(params, i)->valuedouble;
            }
        }

        model_add(model, new_layer);
    }

    cJSON_Delete(json_model);
    return model;
}

#endif // !IO_H