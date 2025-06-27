#ifndef _MLP_H
#define _MLP_H

#include <stdint.h>

typedef struct _mlp_layer {
    uint16_t input_size;
    uint16_t output_size;
    double *weights;
    double *biases;
    double *outputs;
    double *deltas;
} MLPLayer;

typedef enum _forward_func_enum {
    UNKNOWN,
    RELU,
    SIGMOID,
    TANH,
    //TODO: add more support
} ForwardFunc;

typedef struct _mlp {
    uint8_t num_layers;
    MLPLayer *layers;
    double (*func)(double);
} MLP;

void forward(MLP *mlp, double *inp);
void back(); //TODO: add
void create_mlp(MLP *mlp, uint16_t *layer_size, uint8_t num_layers, ForwardFunc func_type);

#endif