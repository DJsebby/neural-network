#include "NeuralNetwork.h"

void NeuralNetwork::add_layer(const DenseLayer& layer) {
  layers.push_back(layer);
}

matrix NeuralNetwork::forward(const matrix& input) {
  matrix output = input;
  for (auto& layer : layers) {
    output = layer.forward(output);
  }
  return output;
}

float NeuralNetwork::computeLoss(const std::vector<float>& predicted,
                                 const std::vector<float>& target) {
  float sum = 0;
  int n = predicted.size();
  for (int i = 0; i < n; ++i) {
    float diff = predicted[i] - target[i];
    sum += diff * diff;
  }
  return sum / n;
}

std::vector<float> NeuralNetwork::computeLossDerivative(
    const std::vector<float>& predicted, const std::vector<float>& target) {
  int n = predicted.size();
  std::vector<float> grad(n);
  for (int i = 0; i < n; ++i) {
    grad[i] = 2 * (predicted[i] - target[i]) / n;
  }
  return grad;
}