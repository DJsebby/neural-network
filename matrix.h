#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "activations.h"

class matrix {
 private:
  std::vector<std::vector<double>> data;

 public:
  matrix();
  matrix(int a, int b);
  void insert_num(int a, int b, int num);
  void insert_matrix(std::vector<std::vector<double>> &mat);
  matrix add(const matrix &other) const;
  matrix subtract(const matrix &other) const;
  matrix multiply(const matrix &other) const;
  matrix inverse() const;
  matrix transpose() const;
  std::string get_dimension_s() const;
  std::vector<int> get_dimension_int() const;
  void print_matrix();
  void scalar_multiply(double num);
  matrix hadamard(const matrix &other) const;
  matrix apply_function(const std::function<double(double)> &func) const;

  void randomise(double min, double max);
  static matrix from_vector(const std::vector<double> &vec);
  std::vector<double> to_vector() const;
  ~matrix() = default;
};
