#pragma once

#include <vector>

#include "layers/denseLayer.h"
#include "matrix.h"

class NeuralNetwork {
 private:
  std::vector<DenseLayer> layers;

 public:
  NeuralNetwork() = default;

  void add_layer(const DenseLayer& layer);

  matrix forward(const matrix& input);

  float computeLoss(const std::vector<float>& predicted,
                    const std::vector<float>& target);
  std::vector<float> computeLossDerivative(const std::vector<float>& predicted,
                                           const std::vector<float>& target);
};
