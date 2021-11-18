#include <stdlib.h>

typedef struct {
    unsigned int size;
    double *weights;
} layer_t;

typedef struct {
    layer_t input;
    layer_t hidden;
    layer_t output;    
} network_t;

layer_t create_layer(unsigned int size) {
    layer_t layer;
    layer.size = size;
    layer.weights = calloc(sizeof(double), size);
    return layer;
}

void destroy_layer(const layer_t *layer) {
    free(layer->weights);
}

network_t create_network(unsigned int input, unsigned int hidden, unsigned int output) {
    network_t network;
    network.input = create_layer(input);
    network.hidden = create_layer(hidden);
    network.output = create_layer(output);
    return network;
}
void destroy_network(const network_t *network) {
    destroy_layer(&(network->input));
    destroy_layer(&(network->hidden));
    destroy_layer(&(network->output));
}

int main() {
    return 0;
}