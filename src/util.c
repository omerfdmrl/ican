#ifndef ITIL_H

#define ITIL_H

#include "ican.h"

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
    return fmaxf(0.0f, x);  // Eğer x negatifse 0 döndür, pozitifse x'i döndür
}

float random_uniform(float low,float high) {
	return ((float) rand() / (float) RAND_MAX) * (high - low) + low;
}
float random_normal(float mean, float stddev) {
    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;

    double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);
    
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

int oldProgress = -1;

void print_progress_header(size_t epoch, size_t total_epochs, float cost) {
    printf("\n----------------------------------------------------------------------------------------------------------------------\n");
    printf("Epoch %zu/%zu - Cost = %.12f\n", epoch + 1, total_epochs, cost);
    printf("----------------------------------------------------------------------------------------------------------------------\n");
}

void print_progress(size_t count, size_t max, float cost) {
    float progress = (float)count / max;
    if(oldProgress == (int)(progress * 100)) return;
    oldProgress = (int)(progress * 100);
    int barWidth = 70;
    printf("\rProgress [");
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) printf("=");
        else if (i == pos) printf(">");
        else printf(" ");
    }
    printf("] %d%% Complete - Cost = %.12f", (int)(progress * 100.0), cost);
    fflush(stdout);
    if (count == max) {
        printf("\n\n");
    }
}

void print_progress_footer(time_t timer, float cost) {
    double time_taken = ((double)timer)/CLOCKS_PER_SEC;
    printf("\n\n----------------------------------------------------------------------------------------------------------------------\n");
    printf("Time Taken = %f - Cost = %.12f\n", time_taken, cost);
    printf("----------------------------------------------------------------------------------------------------------------------\n");
}



Iray2D *csv_read(const char *filename, size_t rows, size_t cols) {
    FILE *file = fopen(filename, "r");
    ASSERT(file != NULL);

    Iray2D *data = iray2d_alloc(rows, cols);

    char line[MAX_LINE_BYTE];
    size_t row = 0;
    while (fgets(line, sizeof(line), file) && row < rows) {
        char *token = strtok(line, ",");
        size_t col = 0;
        while (token != NULL && col < cols) {
            data->data[row][col] = atof(token);
            token = strtok(NULL, ",");
            col++;
        }
        row += 1;
    }

    fclose(file);
    return data;
}

Iray2D *bin_read(const char *filename, size_t rows, size_t cols) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Unable to open file");
        exit(EXIT_FAILURE);
    }

    Iray2D *data = iray2d_alloc(rows, cols);

    for (size_t row = 0; row < rows; row++) {
        size_t read_size = fread(data->data[row], sizeof(float), cols, file);
        if (read_size != cols) {
            fprintf(stderr, "Error reading row %zu\n", row);
            fclose(file);
            exit(EXIT_FAILURE);
        }
    }

    fclose(file);
    return data;
}

#endif // !ITIL_H