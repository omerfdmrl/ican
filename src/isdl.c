#ifndef ISDL_H

#define ISDL_H

#include "ican.h"

ISDLContext *isdl_alloc(int width, int height, bool resizable) {
    ISDLContext *context = malloc(sizeof(ISDLContext));
    ISERT_MSG(context != NULL, "Memory allocation for ISDLContext failed");

    ISERT_MSG(SDL_Init(SDL_INIT_VIDEO) == 0, "Context could not created");

    context->window = SDL_CreateWindow("I See", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, width, height, resizable ? SDL_WINDOW_RESIZABLE : SDL_WINDOW_SHOWN);
    ISERT_MSG(context->window != NULL, "Window could not created");

    context->renderer = SDL_CreateRenderer(context->window, -1, SDL_RENDERER_ACCELERATED);
    ISERT_MSG(context->renderer != NULL, "Rendere could not created");

    return context;
}

void isdl_free(ISDLContext *context) {
    SDL_DestroyRenderer(context->renderer);
    SDL_DestroyWindow(context->window);
    SDL_Quit();
    free(context);
}

void img_show(ISDLContext *context, Iray3D *img) {
    SDL_Surface *surface = SDL_CreateRGBSurfaceWithFormat(0, img->cols, img->rows, 32, SDL_PIXELFORMAT_RGBA32);
    ISERT_MSG(surface != NULL, "Surface could not created");

    for (size_t y = 0; y < img->rows; y++) {
        for (size_t x = 0; x < img->cols; x++) {
            Uint32 pixel;
            if (img->depth == 4) {
                pixel = SDL_MapRGBA(surface->format, img->data[y][x][0], img->data[y][x][1], img->data[y][x][2], img->data[y][x][3]);
            } else if (img->depth == 3) {
                pixel = SDL_MapRGB(surface->format, img->data[y][x][0], img->data[y][x][1], img->data[y][x][2]);
            } else {
                pixel = SDL_MapRGB(surface->format, img->data[y][x][0], img->data[y][x][0], img->data[y][x][0]);
            }
            ((Uint32*)surface->pixels)[y * img->cols + x] = pixel;
        }
    }

    SDL_Texture *texture = SDL_CreateTextureFromSurface(context->renderer, surface);
    ISERT_MSG(texture != NULL, "Texture could not created");

    int quit = 0;
    SDL_Event event;
    while (!quit) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                quit = 1;
            }
        }
        SDL_RenderClear(context->renderer);
        SDL_RenderCopy(context->renderer, texture, NULL, NULL);
        SDL_RenderPresent(context->renderer);
        SDL_Delay(100);
    }

    SDL_FreeSurface(surface);
    SDL_DestroyTexture(texture);
}

void model_show_render_fill_circle(SDL_Renderer *renderer, int cx, int cy, int radius) {
    for (int w = 0; w < radius * 2; w++) {
        for (int h = 0; h < radius * 2; h++) {
            int dx = radius - w;
            int dy = radius - h;
            if ((dx*dx + dy*dy) <= (radius * radius)) {
                SDL_RenderDrawPoint(renderer, cx + dx, cy + dy);
            }
        }
    }
}

void model_show_loss_graph(ISDLContext *context, Iray1D *loss_history, size_t epoch_count, int width, int height) {
    SDL_SetRenderDrawColor(context->renderer, 0, 0, 0, 255);
    SDL_RenderClear(context->renderer);

    SDL_SetRenderDrawColor(context->renderer, 255, 0, 0, 255);

    int margin = 20;
    int graph_height = height - 2 * margin;
    float max_loss = loss_history->data[0];

    for (size_t i = 1; i < epoch_count; i++) {
        if (loss_history->data[i] > max_loss) max_loss = loss_history->data[i];
    }

    if (max_loss == 0) max_loss = 1;

    for (size_t i = 1; i < epoch_count; i++) {
        if(loss_history->data[i] == 0) continue; 
        int x1 = (i - 1) * (width - 2 * margin) / (epoch_count - 1) + margin;
        int y1 = height - margin - (loss_history->data[i - 1] / max_loss * graph_height);
        int x2 = i * (width - 2 * margin) / (epoch_count - 1) + margin;
        int y2 = height - margin - (loss_history->data[i] / max_loss * graph_height);

        SDL_RenderDrawLine(context->renderer, x1, y1, x2, y2);
    }
}

void model_show(ISDLContext *context, Model *model, Iray1D *loss_history) {
    int quit = 0;
    SDL_Event event;
    while (!quit) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                quit = 1;
            }
        }

        SDL_RenderClear(context->renderer);
        
        int width = 0, height = 0;
        SDL_GetWindowSize(context->window, &width, &height);
        int lossWidth = width / 2.4;
        int lossHeight = height / 2.3;
        int neuronRadius = width / 80;
        int startX = neuronRadius + 20;
        int startY = lossHeight + ((height - lossHeight) / 2);
        int layerWidth = width / (2 * model->layer_count);
        model_show_loss_graph(context, loss_history, loss_history->rows, lossWidth, lossHeight);
        for (size_t i = 0; i < model->layer_count; i++) {
            Layer *layer = model->layers[i];
            int layerX = startX + i * layerWidth;
            int neuronYArray[layer->inputSize];

            int layerNeuronCount = layer->inputSize;
            int maxNeuronHeight = height - lossHeight - 40;
            int neuronRadius = width / 80;
            int neuronSpacing = neuronRadius * 2 + 5;
            if (layerNeuronCount * neuronSpacing > maxNeuronHeight) {
                neuronSpacing = maxNeuronHeight / layerNeuronCount;
                neuronRadius = (neuronSpacing - 5) / 2;
            }

            int layerHeight = (layerNeuronCount - 1) * neuronSpacing;
            int layerStartY = lossHeight + ((height - lossHeight) / 2) - layerHeight / 2;

            for (size_t j = 0; j < layerNeuronCount; j++) {
                int neuronY = layerStartY + j * neuronSpacing;
                neuronYArray[j] = neuronY;
                int isActive = layer->output->data[j] > 0;

                SDL_SetRenderDrawColor(context->renderer, isActive ? 0 : 150, isActive ? 255 : 150, 0, 255);
                model_show_render_fill_circle(context->renderer, layerX, neuronY, neuronRadius);
                if (i < model->layer_count - 1) {
                    Layer *nextLayer = model->layers[i + 1];
                    int nextLayerNeuronCount = nextLayer->inputSize;
                    int nextLayerHeight = (nextLayerNeuronCount - 1) * neuronSpacing;
                    int nextLayerStartY = lossHeight + ((height - lossHeight) / 2) - nextLayerHeight / 2;

                    for (size_t k = 0; k < nextLayerNeuronCount; k++) {
                        float weight = (layer->weight != NULL && layer->weight->rows != 0) ? layer->weight->data[j][k] : 0;
                        Uint8 red = weight > 0 ? 0 : (Uint8)(255 * -weight);
                        Uint8 green = weight > 0 ? (Uint8)(255 * weight) : 0;
                        
                        SDL_SetRenderDrawColor(context->renderer, red, green, 0, 150);
                        int nextNeuronY = nextLayerStartY + k * neuronSpacing;
                        SDL_RenderDrawLine(context->renderer, layerX + neuronRadius, neuronY, layerX + layerWidth - neuronRadius, nextNeuronY);
                    }
                }
            }

            if (layer->name == GRU || layer->name == RNN) {
                for (size_t j = 0; j < layer->inputSize; j++) {
                    float weight = layer->weight->data[j][0];

                    Uint8 red, green, blue;
                    if (weight > 0) {
                        red = 0;
                        green = (Uint8)(255 * weight);
                        blue = 0;
                    } else {
                        red = (Uint8)(255 * -weight);
                        green = 0;
                        blue = 0;
                    }
                    SDL_SetRenderDrawColor(context->renderer, red, green, blue, 150);
                    SDL_RenderDrawLine(context->renderer, layerX, neuronYArray[j], layerX, neuronYArray[j]);

                    if (j > 0) {
                        SDL_RenderDrawLine(context->renderer, layerX, neuronYArray[j], 
                                        layerX, neuronYArray[j - 1]);
                    }
                }
            }
        }
        
        SDL_RenderPresent(context->renderer);

        SDL_Delay(100);
    }

}

#endif // ISDL_H