
#include "mnist_loader.h"

#include <filesystem>
#include <iostream>
#include <stdexcept>

#include "stb_image.h"
namespace fs = std::filesystem;
std::vector<std::pair<std::vector<float>, uint8_t>> load_mnist_dataset(
    const std::string& dataset_path) {
  std::vector<std::pair<std::vector<float>, uint8_t>> dataset;

  for (int label = 0; label <= 9; label++) {
    std::string folder = dataset_path + "/" + std::to_string(label);
    if (!fs::exists(folder)) {
      std::cerr << "Warning: Folder not found: " << folder << std::endl;
      continue;
    }

    for (const auto& entry : fs::directory_iterator(folder)) {
      if (!entry.is_regular_file()) continue;

      const std::string& image_path = entry.path().string();
      int width, height, channels;
      unsigned char* data = stbi_load(image_path.c_str(), &width, &height,
                                      &channels, 1);  // force grayscale

      if (!data) {
        std::cerr << "Failed to load: " << image_path << std::endl;
        continue;
      }

      if (width != 28 || height != 28) {
        std::cerr << "Invalid image size in: " << image_path << " (" << width
                  << "x" << height << ")" << std::endl;
        stbi_image_free(data);
        continue;
      }

      std::vector<float> image;
      image.reserve(28 * 28);
      for (int i = 0; i < 28 * 28; ++i) {
        image.push_back(data[i] / 255.0f);  // normalize pixel to [0, 1]
      }

      dataset.emplace_back(image, static_cast<uint8_t>(label));
      stbi_image_free(data);
    }
  }

  return dataset;
}
