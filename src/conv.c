#ifndef CONV_H

#define CONV_H

#include "ican.h"

float CONV_EMBOSS[3][3] = {{-1, -1, 0}, {-1, 0, 1}, {0, -1, -1}};
float CONV_EMBOSS2[3][3] = {{-2, -1, 0}, {-1, 1, 1}, {0, 1, 2}};
float CONV_LAPLACIAN[3][3] = {{1, 1, 1}, {1, -8, 1}, {1, 1, 1}};
float CONV_BLUR[3][3] = {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
float CONV_SOBELX[3][3] = {{1, 0, -2}, {2, 0, -2}, {1, 0, 1}};
float CONV_SOBELY[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};

Iray3D *img_conv(Iray3D *img, float kernel[3][3], size_t stride) {
    size_t new_width = (img->cols - 3) / stride + 1;
    size_t new_height = (img->rows - 3) / stride + 1;

    Iray3D *output = iray3d_alloc(new_height, new_width, img->depth);

    for (size_t i = 0; i < new_height; i++) {
        for (size_t j = 0; j < new_width; j++) {
            Iray1D *sum = iray1d_alloc(img->depth);

            for (size_t ki = 0; ki < 3; ki++) {
                for (size_t kj = 0; kj < 3; kj++) {
                    size_t img_x = j * stride + kj;
                    size_t img_y = i * stride + ki;

                    if (img_x < img->cols && img_y < img->rows) {
                        for (size_t channel = 0; channel < img->depth; channel++) {
                            sum->data[channel] += img->data[img_y][img_x][channel] * kernel[ki][kj];
                        }
                    }
                }
            }

            for (size_t k = 0; k < img->depth; k++) {
                output->data[i][j][k] = sum->data[k];
            }
            iray1d_free(sum);
        }
    }

    return output;
}

Iray3D *img_max_pool(Iray3D* img, size_t pool_size) {
    size_t new_width = img->cols / pool_size;
    size_t new_height = img->rows / pool_size;

    Iray3D* output = iray3d_alloc(new_height, new_width, img->depth);

    for (size_t i = 0; i < new_height; i++) {
        for (size_t j = 0; j < new_width; j++) {
            Iray1D *max_val = iray1d_alloc(img->depth);
            for (size_t k = 0; k < img->depth; k++) {
                max_val->data[k] = -FLT_MAX;
            }

            for (size_t ki = 0; ki < pool_size; ki++) {
                for (size_t kj = 0; kj < pool_size; kj++) {
                    size_t img_x = j * pool_size + kj;
                    size_t img_y = i * pool_size + ki;

                    if (img_x < img->cols && img_y < img->rows) {
                        for (size_t channel = 0; channel < img->depth; channel++) {
                            if (img->data[img_y][img_x][channel] > max_val->data[channel]) {
                                max_val->data[channel] = img->data[img_y][img_x][channel];
                            }
                        }
                    }
                }
            }

            for (size_t channel = 0; channel < img->depth; channel++) {
                output->data[i][j][channel] = max_val->data[channel];
            }
            iray1d_free(max_val);
        }
    }

    return output;
}

Iray3D *img_min_pool(Iray3D* img, size_t pool_size) {
    size_t new_width = img->cols / pool_size;
    size_t new_height = img->rows / pool_size;

    Iray3D* output = iray3d_alloc(new_height, new_width, img->depth);

    for (size_t i = 0; i < new_height; i++) {
        for (size_t j = 0; j < new_width; j++) {
            Iray1D *max_val = iray1d_alloc(img->depth);
            for (size_t k = 0; k < img->depth; k++) {
                max_val->data[k] = FLT_MAX;
            }

            for (size_t ki = 0; ki < pool_size; ki++) {
                for (size_t kj = 0; kj < pool_size; kj++) {
                    size_t img_x = j * pool_size + kj;
                    size_t img_y = i * pool_size + ki;

                    if (img_x < img->cols && img_y < img->rows) {
                        for (size_t channel = 0; channel < img->depth; channel++) {
                            if (img->data[img_y][img_x][channel] < max_val->data[channel]) {
                                max_val->data[channel] = img->data[img_y][img_x][channel];
                            }
                        }
                    }
                }
            }

            for (size_t channel = 0; channel < img->depth; channel++) {
                output->data[i][j][channel] = max_val->data[channel];
            }
            iray1d_free(max_val);
        }
    }

    return output;
}

Iray3D *img_mean_pool(Iray3D* img, size_t pool_size) {
    size_t new_width = img->cols / pool_size;
    size_t new_height = img->rows / pool_size;

    Iray3D* output = iray3d_alloc(new_height, new_width, img->depth);
    ASSERT_MSG(output != NULL, "Output image allocation failed");

    for (size_t i = 0; i < new_height; i++) {
        for (size_t j = 0; j < new_width; j++) {
            Iray1D *sum_val = iray1d_alloc(img->depth);
            for (size_t k = 0; k < img->depth; k++) {
                sum_val->data[k] = 0.0f;
            }

            for (size_t ki = 0; ki < pool_size; ki++) {
                for (size_t kj = 0; kj < pool_size; kj++) {
                    size_t img_x = j * pool_size + kj;
                    size_t img_y = i * pool_size + ki;

                    if (img_x < img->cols && img_y < img->rows) {
                        for (size_t channel = 0; channel < img->depth; channel++) {
                            sum_val->data[channel] += img->data[img_y][img_x][channel];
                        }
                    }
                }
            }

            for (size_t channel = 0; channel < img->depth; channel++) {
                output->data[i][j][channel] = sum_val->data[channel] / (pool_size * pool_size);
            }
            iray1d_free(sum_val);
        }
    }

    return output;
}

Iray3D *img_edge_detect(Iray3D *img) {
    Iray3D *grad_x = img_conv(img, CONV_SOBELX, 1);
    Iray3D *grad_y = img_conv(img, CONV_SOBELY, 1);

    size_t new_width = grad_x->cols;
    size_t new_height = grad_x->rows;
    Iray3D *output = iray3d_alloc(new_height, new_width, img->depth);

    for (size_t i = 0; i < new_height; i++) {
        for (size_t j = 0; j < new_width; j++) {

            float gx = grad_x->data[i][j][0];
            float gy = grad_y->data[i][j][0];
            float magnitude = sqrtf(gx * gx + gy * gy);
            float val = fminf(fmaxf(magnitude, 0), 255);
            for (size_t k = 0; k < img->depth; k++) {
                output->data[i][j][k] = val;
                if(k == 3) {
                    output->data[i][j][k] = 255;
                }
            }
            
        }
    }

    iray3d_free(grad_x);
    iray3d_free(grad_y);

    return output;
}

#endif // !CONV_H