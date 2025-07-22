#include "NeuralNetwork.h"
#include "mnist_loader/mnist_loader.h"

int main() {
  NeuralNetwork network;
  // load in the images
  std::vector<std::pair<std::vector<double>, std::vector<double>>> dataset =
      load_mnist_dataset("/dataset");

  // make the dataset usable
  std::vector<matrix> train_inputs;
  std::vector<std::vector<double>> train_targets;
  for (const auto& [input_vec, target_vec] : dataset) {
    matrix input_matrix;
    input_matrix.from_vector(input_vec);
    train_inputs.push_back(input_matrix);
    train_targets.push_back(target_vec);
  }

  // define the network
  DenseLayer layer1(784, 128, sigmoid);
  DenseLayer layer2(128, 10, sigmoid);
  std::vector<DenseLayer> model = {layer1, layer2};
  network.set_layers(model);
  // training parameters
  int epochs = 10;
  double learningRate = 0.1;

  // train te model
  network.train(train_inputs, train_targets, learningRate, epochs);

  return 0;
}