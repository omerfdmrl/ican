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

#endif // !PROCESS_H