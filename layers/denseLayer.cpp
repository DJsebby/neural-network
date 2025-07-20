#include "denseLayer.h"

DenseLayer::DenseLayer(int inputSize, int outputSize,
                       std::function<double(double)> activationFunc)
    : weights(inputSize, outputSize),
      biases(1, outputSize),
      activation(activationFunc) {
  weights.randomise(-1, 1);
  biases.randomise(-1, 1);
}
matrix DenseLayer::forward(const matrix& input) {
  matrix z = input.multiply(weights);
  z = z.add(biases);
  return z.apply_function(activation);
}

matrix DenseLayer::get_weights() const { return weights; }

matrix DenseLayer::get_biases() const { return biases; }