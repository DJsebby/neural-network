#pragma once

#include <vector>

#include "layers/denseLayer.h"
#include "matrix.h"

class NeuralNetwork {
 private:
  std::vector<DenseLayer> layers;

 public:
  NeuralNetwork() = default;

  void set_layers(std::vector<DenseLayer>& new_layers);

  void add_layer(const DenseLayer& layer);

  matrix forward(const matrix& input);

  void train(const std::vector<matrix>& inputs,
             const std::vector<std::vector<double>>& targets,
             double learning_rate, int epochs);

  double computeLoss(const std::vector<double>& predicted,
                     const std::vector<double>& target);
  std::vector<double> computeLossDerivative(
      const std::vector<double>& predicted, const std::vector<double>& target);
};
