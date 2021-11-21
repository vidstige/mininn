#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h> // for strtok

double randfrom(double min, double max) 
{
    double range = (max - min); 
    double div = RAND_MAX / range;
    return min + (rand() / div);
}

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

array_t create_array_like(array_t like) {
    return create_array(like.size);
}

void array_copy(array_t to, array_t from) {    
    // TODO: assert to.size <= from.size
    for (size_t i = 0; i < to.size; i++) {
        to.values[i] = from.values[i];
    }
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
    double *bias;
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
    layer.bias = calloc(sizeof(double), output_size);
    
    for (size_t i = 0; i < output_size; i++) {
        layer.weights[i] = calloc(sizeof(double), input_size);
    }
    return layer;
}

double activation(const layer_t *layer, array_t input, size_t n) {
    assert(n < layer->output_size);
    assert(layer->input_size == input.size);

    double *weights = layer->weights[n];
	double a = layer->bias[n];
    for (size_t i = 0; i < layer->input_size; i++) {
        a += weights[i] * input.values[i];
    }
	return a;
}

void forward_layer_to(const layer_t *layer, array_t input, array_t output) {
    assert(layer->input_size == input.size);
    assert(layer->output_size == output.size);
    for (size_t i = 0; i < layer->output_size; i++) {
        output.values[i] = sigmoid(activation(layer, input, i));
    }
}

void destroy_layer(const layer_t *layer) {
    for (size_t i = 0; i < layer->output_size; i++) {
        free(layer->weights[i]);
    }
    free(layer->weights);
    free(layer->bias);
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

void update_weights(layer_t *layer, array_t input, double *deltas, double learning_rate) {
    assert(layer->input_size == input.size);
    // TODO: deltas.size == layer->output_size
    for (size_t i = 0; i < layer->output_size; i++) {
        for (size_t j = 0; j < layer->input_size; j++) {
            layer->weights[i][j] -= learning_rate * deltas[i] * input.values[i];
        }
        layer->bias[i] -= learning_rate * deltas[i];
    }
}

void backpropagate(network_t *network, array_t input, array_t expected, double learning_rate) {
    // forward each layer and remember outputs
    array_t hidden_output = create_array(network->hidden.output_size);
    array_t output = create_array(network->output.output_size);
    forward_layer_to(&(network->hidden), input, hidden_output);
    forward_layer_to(&(network->output), hidden_output, output);

    // calculate deltas for output layer
    double output_deltas[network->output.output_size];
    for (size_t i = 0; i < network->output.output_size; i++) {
        const double error = output.values[i] - expected.values[i];
        output_deltas[i] = error * sigmoid_derivative(output.values[i]);
    }

    // calculate deltas for hidden layer
    double hidden_deltas[network->hidden.output_size];
    for (size_t i = 0; i < network->hidden.output_size; i++) {
        double error = 0.0;
        for (size_t j = 0; j < network->output.output_size; j++) {
            error += network->output.weights[j][i] * output_deltas[j];
        }
        hidden_deltas[i] = error * sigmoid_derivative(hidden_output.values[i]);
    }

    // Update weights in the hidden layer
    update_weights(&(network->hidden), input, hidden_deltas, learning_rate);

    // Update weight in the output layer. Input = hidden_output
    update_weights(&(network->output), hidden_output, output_deltas, learning_rate);

    destroy_array(&hidden_output);
    destroy_array(&output);
}

void destroy_network(const network_t *network) {
    destroy_layer(&(network->hidden));
    destroy_layer(&(network->output));
}

void one_hot(array_t array, int n) {
    for (size_t i = 0; i < array.size; i++) {
        array.values[i] = 0.0;
    }
    array.values[n] = 1.0;
}

typedef struct {
    size_t size;
    size_t allocated;
    array_t *rows;
} dataset_t;

dataset_t load_dataset(FILE* file, size_t columns) {
    dataset_t dataset;
    dataset.size = 0;
    dataset.allocated = 0;
    dataset.rows = malloc(0);

    size_t i = 0;
    char *line = NULL;
    size_t size;
    while (getline(&line, &size, file) != -1) {
        if (i >= dataset.allocated) {
            dataset.allocated += 20;
            dataset.rows = realloc(dataset.rows, sizeof(array_t) * dataset.allocated);
        }

        size_t j = 0;
        dataset.rows[i] = create_array(columns);
        for (char *tok = strtok(line, ","); tok && *tok; tok = strtok(NULL, ",")) {
            dataset.rows[i].values[j] = strtod(tok, NULL);
            j++;
        }
        i++;
    }
    dataset.size = i;
    return dataset;
}

void destroy_dataset(const dataset_t *dataset) {
    for (size_t i = 0; i < dataset->size; i++) {
        destroy_array(&(dataset->rows[i]));
    }
    free(dataset->rows);
}

// computes squared error
double error2_for(const network_t *network, array_t input, array_t expected) {
    array_t output = create_array_like(expected);
    forward_to(network, input, output);
    double sum = 0.0;
    for (size_t i = 0; i < output.size; i++) {
        sum += (output.values[i] - expected.values[i]) * (output.values[i] - expected.values[i]);
    }
    destroy_array(&output);
    return sum;
}

void initialize_layer(layer_t *layer) {
    for (size_t i = 0; i < layer->output_size; i++) {
        for (size_t j = 0; j < layer->input_size; j++) {
            layer->weights[i][j] = randfrom(0.0, 1.0);
        }
    }
}

void initialize_network(network_t *network) {
    initialize_layer(&(network->hidden));
    initialize_layer(&(network->output));
}

int main() {
    network_t network = create_network(2, 1, 2);
    initialize_network(&network);

    dataset_t toy = load_dataset(stdin, 4);
    
    const double learning_rate = 0.5;
    array_t input = create_array(2);
    array_t expected = create_array(2);
    for (size_t epoch = 0; epoch < 40; epoch++) {
        double error2 = error2_for(&network, input, expected);
        for (size_t i = 0; i < toy.size; i++) {
            array_copy(input, toy.rows[i]);
            one_hot(expected, (int)toy.rows[i].values[2]);
            backpropagate(&network, input, expected, learning_rate);

            error2 += error2_for(&network, input, expected);
        }
        printf("> epoch=%ld. error2=%f\n", epoch, error2);
    }
    destroy_array(&input);
    destroy_array(&expected);

    destroy_dataset(&toy);
    destroy_network(&network);
    return 0;
}
