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


#endif // ISDL_H