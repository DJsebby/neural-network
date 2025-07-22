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

double NeuralNetwork::computeLoss(const std::vector<double>& predicted,
                                  const std::vector<double>& target) {
  double sum = 0;
  int n = predicted.size();
  for (int i = 0; i < n; ++i) {
    double diff = predicted[i] - target[i];
    sum += diff * diff;
  }
  return sum / n;
}

std::vector<double> NeuralNetwork::computeLossDerivative(
    const std::vector<double>& predicted, const std::vector<double>& target) {
  int n = predicted.size();
  std::vector<double> grad(n);
  for (int i = 0; i < n; ++i) {
    grad[i] = 2 * (predicted[i] - target[i]) / n;
  }
  return grad;
}

void NeuralNetwork::train(const std::vector<matrix>& inputs,
                          const std::vector<std::vector<double>>& targets,
                          double learning_rate, int epochs) {
  for (int epoch = 0; epoch < epochs; ++epoch) {
    double total_loss = 0.0;

    for (size_t i = 0; i < inputs.size(); ++i) {
      // Forward pass
      matrix prediction = forward(inputs[i]);

      // Compute loss
      std::vector<double> predicted_values = prediction.to_vector();
      total_loss += computeLoss(predicted_values, targets[i]);

      // Compute loss gradient
      std::vector<std::vector<double>> dL_dy = {
          computeLossDerivative(predicted_values, targets[i])};

      // Convert gradient to a matrix (assuming row vector shape)
      matrix grad_output;
      grad_output.insert_matrix({dL_dy});  // 1-row matrix

      // Backward pass
      for (int l = layers.size() - 1; l >= 0; --l) {
        grad_output = layers[l].backwards(grad_output, learning_rate);
      }
    }

    std::cout << "Epoch " << epoch + 1 << "/" << epochs
              << ", Loss: " << total_loss / inputs.size() << std::endl;
  }
}

void NeuralNetwork::set_layers(std::vector<DenseLayer>& new_layers) {
  layers = new_layers;
}