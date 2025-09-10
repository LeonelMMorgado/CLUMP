#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "mlp.h"

double randd() {
    return (double)rand()/RAND_MAX;
}

void init_layer(MLPLayer *layer, uint16_t inp_size, uint16_t out_size) {
    layer->input_size;
    layer->output_size;
    layer->weights = malloc(sizeof(double) * inp_size * out_size);
    layer->biases = calloc(out_size, sizeof(double));
    layer->outputs = calloc(out_size, sizeof(double));
    layer->deltas = calloc(out_size, sizeof(double));
    for(int i = 0; i < inp_size * out_size; i++)
        layer->weights[i] = randd() * 0.5;
}

double relu(double x) {
    return x > 0 ? x : 0;
}

double sigmoid(double x) {
    double etx = exp(x);//e^x
    double oneup = etx+1;
    return etx/oneup;
}

void create_mlp(MLP *mlp, uint16_t *layer_size, uint8_t num_layers, ForwardFunction function_type) {
    mlp->num_layers = num_layers - 1;
    mlp->layers = malloc(sizeof(MLPLayer) * num_layers);
    if(!mlp->layers) return;
    for(uint8_t i = 0; i < mlp->num_layers; i++)
        init_layer(&mlp->layers[i], layer_size[i], layer_size[i + 1]);
    switch(function_type) {
        case RELU:
            mlp->func = &relu;
            break;
        case SIGMOID:
            mlp->func = &sigmoid;
            break;
        case TANH:
            mlp->func = &tanh;
            break;
    }
}