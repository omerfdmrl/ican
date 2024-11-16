#ifndef PROCESS_H

#define PROCESS_H

#include "ican.h"

Iray2D *one_hot_encoding(Iray2D *data) {
    size_t category_length = 0;
    float *categories = NULL;
    Iray2D *temp;

    for (size_t i = 0; i < data->rows; i++) {
        for (size_t j = 0; j < data->cols; j++) {
            bool exists = false;
            for (size_t k = 0; k < category_length; k++) {
                if (categories[k] == data->data[i][j]) {
                    exists = true;
                    break;
                }
            }
            if (!exists) {
                categories = (float *)realloc(categories, (category_length + 1) * sizeof(float));
                categories[category_length] = data->data[i][j];
                category_length++;
            }
        }
    }

    temp = iray2d_alloc(data->rows, category_length);

    for (size_t i = 0; i < data->rows; i++) {
        for (size_t j = 0; j < data->cols; j++) {
            for (size_t k = 0; k < category_length; k++) {
                if (categories[k] == data->data[i][j]) {
                    temp->data[i][k] = 1;
                } else {
                    temp->data[i][k] = 0;
                }
            }
        }
    }
    return temp;
}

Iray2D *standard_scaler(Iray2D *data) {
    Iray2D *scaledData = iray2d_alloc(data->rows, data->cols);

    for (int j = 0; j < data->cols; j++) {
        double sum = 0.0;
        for (int i = 0; i < data->rows; i++) {
            sum += data->data[i][j];
        }
        double mean = sum / data->rows;

        double stddev_sum = 0.0;
        for (int i = 0; i < data->rows; i++) {
            stddev_sum += (data->data[i][j] - mean) * (data->data[i][j] - mean);
        }
        double stddev = sqrt(stddev_sum / data->rows);

        for (int i = 0; i < data->rows; i++) {
            if (stddev != 0) {
                scaledData->data[i][j] = (data->data[i][j] - mean) / stddev;
            } else {
                scaledData->data[i][j] = data->data[i][j];
            }
        }
    }
    return scaledData;
}

Iray2D *pad_sequences(Iray2D *data, size_t maxLength, PadTypes padding, PadTypes truncating, float value) {
    Iray2D *padded_data_alloc = iray2d_alloc(data->rows, maxLength);
    Iray2D *padded_data = iray2d_fill(padded_data_alloc, value);
    iray2d_free(padded_data_alloc);
    
    for (size_t i = 0; i < data->rows; i++) {
        size_t start_index = 0;
        size_t copy_length = data->cols;
        
        if (data->cols > maxLength) {
            copy_length = maxLength;
            if (truncating == PAD_PRE) {
                start_index = data->cols - maxLength;
            }
        }
        
        size_t pad_start = 0;
        if (padding == PAD_POST) {
            pad_start = 0;
        } else {
            pad_start = maxLength - copy_length;
        }
        
        for (size_t j = 0; j < copy_length; j++) {
            padded_data->data[i][pad_start + j] = data->data[i][start_index + j];
        }
    }
    
    return padded_data;
}

Tokenizer *fit_on_texts(const char *text) {
    Tokenizer *tokenizer = (Tokenizer *)malloc(sizeof(Tokenizer));
    tokenizer->size = 0;
    tokenizer->capacity = 2;
    tokenizer->words = (char **)malloc(tokenizer->capacity * sizeof(char *));

    char *token;
    char temp_str[strlen(text) + 1];
    strcpy(temp_str, text);
    
    token = strtok(temp_str, " ");
    while (token != NULL) {
        if (tokenizer->size >= tokenizer->capacity) {
            tokenizer->capacity *= 2;
            tokenizer->words = (char **)realloc(tokenizer->words, tokenizer->capacity * sizeof(char *));
        }

        int found = 0;
        for (size_t i = 0; i < tokenizer->size; i++) {
            if (strcmp(tokenizer->words[i], token) == 0) {
                found = 1;
                break;
            }
        }

        if (!found) {
            size_t word_len = strlen(token);
            tokenizer->words[tokenizer->size] = (char *)malloc((word_len + 1) * sizeof(char));
            strcpy(tokenizer->words[tokenizer->size], token);
            tokenizer->size++;
        }

        token = strtok(NULL, " ");
    }

    return tokenizer;
}

Iray1D *texts_to_sequences(Tokenizer *tokenizer, const char *text) {
    if (tokenizer == NULL || text == NULL) {
        return NULL;
    }

    char cop_data[strlen(text) + 1];
    strcpy(cop_data, text);
    char *token2 = strtok(cop_data, " ");
    int sayac = 0;

    while (token2 != NULL) {
        sayac++;
        token2 = strtok(NULL, " ");
    }

    float *temp_data = (float *)malloc(sayac * sizeof(float));
    if (temp_data == NULL) {
        return NULL;
    }
    
    size_t count = 0;

    char *token;
    char temp_str[strlen(text) + 1];
    strcpy(temp_str, text);

    token = strtok(temp_str, " ");
    while (token != NULL) {
        for (size_t i = 0; i < tokenizer->size; i++) {
            if (strcmp(tokenizer->words[i], token) == 0) {
                if (count < sayac) {
                    temp_data[count] = (float)(i + 1);
                    count++;
                }
                break;
            }
        }

        token = strtok(NULL, " ");
    }

    Iray1D *data = iray1d_alloc(count);
    if (data == NULL) {
        free(temp_data);
        return NULL;
    }

    for (size_t i = 0; i < count; i++) {
        data->data[i] = temp_data[i];
    }

    free(temp_data);

    return data;
}

void tokenizer_free(Tokenizer *tokenizer) {
    if(tokenizer != NULL) {
        if(tokenizer->words != NULL) {
            free(tokenizer->words);
        }
        free(tokenizer);
    }
}

#endif // !PROCESS_H