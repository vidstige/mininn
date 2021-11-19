#include <stdio.h>
#include <stdlib.h>

typedef struct {
    double *values;
    size_t size;    
} array_t;

array_t create_array(size_t size) {
    array_t array;
    array.size = size;
    array.values = calloc(sizeof(double), size);
    return array;
}
void destroy_array(const array_t *array) {
    free(array->values);
}

typedef struct {
    size_t input_size;
    size_t output_size;
    double **weights;
} layer_t;

typedef struct {
    layer_t hidden;
    layer_t output;    
} network_t;

layer_t create_layer(size_t input_size, size_t output_size) {
    layer_t layer;
    layer.input_size = input_size;
    layer.output_size = output_size;
    layer.weights = calloc(sizeof(double*), input_size);
    
    for (size_t i = 0; i < input_size; i++) {
        layer.weights[i] = calloc(sizeof(double), output_size);
    }
    return layer;
}

void forward_layer_to(const layer_t *layer, array_t input, array_t output) {
    if (layer->input_size != input.size) {
        fprintf(stderr, "Input size %ld does not match layer size %ld\n", input.size, layer->input_size);
    }
}

void destroy_layer(const layer_t *layer) {
    for (size_t i = 0; i < layer->input_size; i++) {
        free(layer->weights[i]);
    }
    free(layer->weights);
}

network_t create_network(size_t input, size_t hidden, size_t output) {
    network_t network;
    network.hidden = create_layer(input, hidden);
    network.output = create_layer(hidden, output);
    return network;
}

void forward_to(const network_t *network, array_t input, array_t output) {
    forward_layer_to(&(network->hidden), input, output);
}

void destroy_network(const network_t *network) {
    destroy_layer(&(network->hidden));
    destroy_layer(&(network->output));
}

int main() {
    network_t network = create_network(2, 1, 2);
    array_t input = create_array(2);
    array_t output = create_array(2);
    
    forward_to(&network, input, output);
    
    destroy_array(&input);
    destroy_array(&output);
    destroy_network(&network);
    return 0;
}
