#include "activations.h"

#include <cmath>

double relu(double x) { return x < 0 ? 0 : x; }
double sigmoid(double x) { return 1 / (1 + std::exp(-x)); }
double tanh(double x) { return std::tanh(x); }

double sigmoid_derivative(double x) { return sigmoid(x) * (1 - sigmoid(x)); }