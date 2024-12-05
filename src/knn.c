#include "machine_learning/supervised_learning/knn.h"

float most_frequent_label(Array1D *labels) {
    int max_label = -2147483648;
    int min_label = 2147483647;

    for (size_t i = 0; i < labels->rows; i++) {
        if (labels->data[i] > max_label) {
            max_label = labels->data[i];
        }
        if (labels->data[i] < min_label) {
            min_label = labels->data[i];
        }
    }

    int label_range = max_label - min_label + 1;
    int *freq = (int*)ICAN_MALLOC(sizeof(int) * label_range);
    for (int i = 0; i < label_range; i++) {
        freq[i] = 0;
    }

    for (size_t i = 0; i < labels->rows; i++) {
        int label = labels->data[i] - min_label;
        freq[label]++;
    }

    int max_freq = 0;
    int most_frequent = -1;
    for (int i = 0; i < label_range; i++) {
        if (freq[i] > max_freq) {
            max_freq = freq[i];
            most_frequent = i + min_label;
        }
    }

    free(freq);
    return (float) most_frequent;
}

KNN *knn_alloc(Array2D *x, Array2D *y, int16 k) {
    KNN *knn = (KNN *)ICAN_MALLOC(sizeof(KNN));
    knn->x = x;
    knn->y = y;
    knn->k = k;
    return knn;
}

void knn_free(KNN *knn) {
    ICAN_FREE(knn);
}

float pow2(float x) {
    return powf(x, 2);
}

float euclidean_distance(float *x, Array1D *y, int64 cols) {
    Array1D *x_data = array1d_from(x, cols);
    array1d_sub_inplace(x_data, y);
    array1d_apply_inplace(x_data, pow2);
    float sum = array1d_sum(x_data);
    array1d_free(x_data);
    return sqrtf(sum);
}

int sort(float *a, float *b, int64 cols) {
    (void) cols;
    return a[0] - b[0];
}

Array2D *calculate_distances(Array2D *dataset, Array2D *target, Array1D *x) {
    Array2D *distances = array2d_alloc(dataset->rows, target->cols + 1);
    for (size_t i = 0; i < dataset->rows; i++) {
        float distance = euclidean_distance(&dataset->data[i * dataset->cols], x, dataset->cols);
        distances->data[i * distances->cols] = distance;
        for (size_t t = 0; t < target->cols; t++) {
            distances->data[i * distances->cols + 1 + t] = target->data[i * target->cols + t];
        }
    }
    array2d_sort_inplace(distances, sort);
    return distances;
}

float knn_predict(KNN *knn, Array1D *x) {
    Array2D *distances = calculate_distances(knn->x, knn->y, x);
    Array2D *labels = array2d_slice(distances, 0, knn->k, 1, distances->cols);
    Array1D *labels1d = array2d_to_array1d(labels);
    array2d_free(distances);
    array2d_free(labels);
    float msl = most_frequent_label(labels1d);
    array1d_free(labels1d);
    return msl;
}