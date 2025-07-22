#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include <string>
#include <utility>
#include <vector>

std::vector<std::pair<std::vector<double>, std::vector<double>>>
load_mnist_dataset(const std::string& dataset_path);

#endif
