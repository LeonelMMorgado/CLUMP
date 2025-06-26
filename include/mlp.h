#ifndef _MLP_H
#define _MLP_H

#include <stdint.h>

typedef struct _mlp_layer {
    uint16_t input_size;
    uint16_t output_size;
    float *weights;
    float *biases;
    float *outputs;
    float *deltas;
} MLPLayer;

typedef enum _forward_func_enum {
    UNKNOWN,
    RELU,
    SIGMOID,
    //TODO: add more support
} ForwardFunc;

typedef struct _mlp {
    uint8_t num_layers;
    MLPLayer *layers;
    ForwardFunc forward_func;
} MLP;

void forward(MLP *mlp, float *inp);
void backward(); //TODO: add
void create_mlp(MLP *mlp, uint16_t *layer_size, uint8_t num_layers);

#endif