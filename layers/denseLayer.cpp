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
  lastInput = input;
  matrix z = input.multiply(weights);
  z = z.add(biases);
  return z.apply_function(activation);
}

matrix DenseLayer::backwards(const matrix& dL_dOutput, double learning_rate) {
  matrix input = lastInput;
  // z = input * weights + biases
  matrix z = input.multiply(weights).add(biases);

  // activation derivative at z, element-wise
  matrix activation_derivative = z.apply_function(
      sigmoid_derivative);  // you need to implement activation_prime

  // 3. element-wise multiply dL_dOutput and activation derivative to get dL_dz
  matrix dL_dz = dL_dOutput.hadamard(activation_derivative);

  // 4. Calculate gradients w.r.t weights and biases
  matrix dL_dW = input.transpose().multiply(dL_dz);
  matrix dL_db = dL_dz.sum_rows();  // or sum over batch dimension

  // 5. Calculate gradient w.r.t input to propagate backwards
  matrix dL_dInput = dL_dz.multiply(weights.transpose());

  // 6. Update weights and biases
  dL_dW.scalar_multiply(learning_rate);
  weights = weights.subtract(dL_dW);
  dL_db.scalar_multiply(learning_rate);
  biases = biases.subtract(dL_db);

  // 7. Return gradient w.r.t input for next layer
  return dL_dInput;
}

matrix DenseLayer::get_weights() const { return weights; }

matrix DenseLayer::get_biases() const { return biases; }