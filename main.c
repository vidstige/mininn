#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double sigmoid(double x) {
	return 1.0 / (1.0 + exp(-x));
}
double sigmoid_derivative(double x) {
	return x * (1.0 - x);
}

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

void print_array(array_t array) {
    printf("[");
    if (array.size > 0) {
        printf("%f", array.values[0]);
    }
    for (size_t i = 1; i < array.size; i++) {
        printf(", %f", array.values[i]);
    }
    printf("]\n");
}

void destroy_array(const array_t *array) {
    free(array->values);
}

typedef struct {
    size_t input_size;
    size_t output_size;
    double **weights;
    double bias;
} layer_t;

typedef struct {
    layer_t hidden;
    layer_t output;    
} network_t;

layer_t create_layer(size_t input_size, size_t output_size) {
    layer_t layer;
    layer.input_size = input_size;
    layer.output_size = output_size;
    layer.weights = calloc(sizeof(double*), output_size);
    
    for (size_t i = 0; i < output_size; i++) {
        layer.weights[i] = calloc(sizeof(double), input_size);
    }
    return layer;
}

double activation(const layer_t *layer, array_t input, size_t n) {
    // TODO: assert n is smaller than layer.outpåutsize
    if (layer->input_size != input.size) {
        fprintf(stderr, "Input size %ld does not match layer size %ld\n", input.size, layer->input_size);
    }

    double *weights = layer->weights[n];
	double a = layer->bias;
    for (size_t i = 0; i < layer->input_size; i++) {
        a += weights[i] * input.values[i];
    }
	return a;
}

void forward_layer_to(const layer_t *layer, array_t input, array_t output) {
    if (layer->input_size != input.size) {
        fprintf(stderr, "Input size %ld does not match layer size %ld\n", input.size, layer->input_size);
    }
    if (layer->output_size != output.size) {
        fprintf(stderr, "Output size %ld does not match layer size %ld\n", output.size, layer->output_size);
    }

    for (size_t i = 0; i < layer->output_size; i++) {
        output.values[i] = sigmoid(activation(layer, input, i));
    }
}

void destroy_layer(const layer_t *layer) {
    for (size_t i = 0; i < layer->output_size; i++) {
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
    array_t tmp = create_array(network->hidden.output_size);
    forward_layer_to(&(network->hidden), input, tmp);
    forward_layer_to(&(network->output), tmp, output);
    destroy_array(&tmp);
}

void destroy_network(const network_t *network) {
    destroy_layer(&(network->hidden));
    destroy_layer(&(network->output));
}

int main() {
    network_t network = create_network(2, 1, 2);
    array_t input = create_array(2);
    array_t output = create_array(2);
    
    print_array(input);
    forward_to(&network, input, output);
    print_array(output);
    
    destroy_array(&input);
    destroy_array(&output);
    destroy_network(&network);
    return 0;
}
