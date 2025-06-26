#include <stdlib.h>
#include <stdint.h>
#include "mlp.h"

float randf() {
    return rand()/RAND_MAX;
}

void init_layer(MLPLayer *layer, uint16_t inp_size, uint16_t out_size) {
    layer->input_size;
    layer->output_size;
    layer->weights = malloc(sizeof(float) * inp_size * out_size);
    layer->biases = calloc(out_size, sizeof(float));
    layer->outputs = calloc(out_size, sizeof(float));
    layer->deltas = calloc(out_size, sizeof(float));
    for(int i = 0; i < inp_size * out_size; i++)
        layer->weights[i] = randf() * 0.5;
}

void create_mlp(MLP *mlp, uint16_t *layer_size, uint8_t num_layers, ForwardFunction function_type) {
    mlp->num_layers = num_layers - 1;
    mlp->layers = malloc(sizeof(MLPLayer) * num_layers);
    for(uint8_t i = 0; i < mlp->num_layers; i++)
        init_layer(&mlp->layers[i], layer_size[i], layer_size[i + 1]);
    mlp->forward_func = function_type;
}