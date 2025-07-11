#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#define STB_IMAGE_IMPLEMENTATION
#include <string>
#include <utility>
#include <vector>

#include "stb_image.h"

std::vector<std::pair<std::vector<float>, uint8_t>> load_mnist_dataset(
    const std::string& dataset_path);

#endif
