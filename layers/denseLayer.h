#include <functional>

#include "../activations/activations.h"
#include "../matrix.h"

#pragma once

class DenseLayer {
 private:
  matrix weights;  // shape: (input_size × output_size)
  matrix biases;   // shape: (1 × output_size)
  std::function<double(double)> activation;

 public:
  DenseLayer(int inputSize, int outputSize,
             std::function<double(double)> activationFunc);
  matrix forward(const matrix& input);

  matrix get_weights() const;
  matrix get_biases() const;
};
