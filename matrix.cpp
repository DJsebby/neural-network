#include "matrix.h"
using namespace std;

matrix::matrix() {}

matrix::matrix(int a, int b) {
  data.resize(a);

  for (size_t i = 0; i < a; i++) {
    data[i].resize(b, 0);
  }
}

string matrix::get_dimension_s() const {
  if (data.size() == 0) {
    return "no matrix";
  }

  string dim = "";
  string a = to_string(data.size());
  string b = to_string(data[0].size());

  dim = a + " x " + b;
  return dim;
}

vector<int> matrix::get_dimension_int() const {
  if (data.size() == 0) {
    return {};
  }

  vector<int> dim;
  dim.push_back(data.size());
  dim.push_back(data[0].size());
  return dim;
}

matrix matrix::add(const matrix &other) const {
  if (this->get_dimension_int() != other.get_dimension_int()) {
    cout << "matrix dimensions don't match\n";
    return matrix();
  }

  vector<int> dim = other.get_dimension_int();
  size_t row = dim[0];
  size_t col = dim[1];

  matrix res(row, col);
  for (size_t i = 0; i < row; i++) {
    for (size_t j = 0; j < col; j++) {
      res.data[i][j] = this->data[i][j] + other.data[i][j];
    }
  }
  return res;
}

matrix matrix::subtract(const matrix &other) const {
  if (this->get_dimension_int() != other.get_dimension_int()) {
    cout << "matrix dimensions don't match\n";
    return matrix();
  }

  vector<int> dim = other.get_dimension_int();
  size_t row = dim[0];
  size_t col = dim[1];

  matrix res(row, col);
  for (size_t i = 0; i < row; i++) {
    for (size_t j = 0; j < col; j++) {
      res.data[i][j] = this->data[i][j] - other.data[i][j];
    }
  }
  return res;
}

matrix matrix::multiply(const matrix &other) const {
  if (this->get_dimension_int()[1] != other.get_dimension_int()[0]) {
    cout << "matrix dimensions aren't compaitible for matrix multiplication\n";
    return matrix();
  }

  vector<int> this_dim = this->get_dimension_int();
  size_t this_row = this_dim[0];
  size_t this_col = this_dim[1];

  vector<int> other_dim = other.get_dimension_int();
  size_t other_row = other_dim[0];
  size_t other_col = other_dim[1];

  matrix res(this_row, other_col);

  //   to get the dot prod

  for (size_t i = 0; i < this_row; ++i) {
    for (size_t j = 0; j < other_col; ++j) {
      double dot_prod = 0;
      for (size_t k = 0; k < this_col; ++k) {
        dot_prod += this->data[i][k] * other.data[k][j];
      }
      res.data[i][j] = dot_prod;
    }
  }
  return res;
}

matrix matrix::inverse() const {
  vector<int> dim = this->get_dimension_int();
  // check if matrix is  square
  if (dim[0] != dim[1]) {
    cout << "matrix is not square";
    return matrix();
  }

  matrix A = *this;
  matrix I = matrix(dim[0], dim[1]);

  // intialising identity matrix
  for (size_t i = 0; i < dim[0]; i++) {
    I.data[i][i] = 1;
  }

  int n = dim[0];
  // gauss jordan elimination
  for (int i = 0; i < n; ++i) {
    // Find pivot
    double pivot = A.data[i][i];
    int pivotRow = i;
    for (int r = i + 1; r < n; r++) {
      if (std::abs(A.data[r][i]) > std::abs(pivot)) {
        pivot = A.data[r][i];
        pivotRow = r;
      }
    }

    if (std::abs(pivot) < 1e-12) {
      std::cout << "Matrix is singular.\n";
      return matrix();
    }

    // swap rows in A and I if needed
    if (pivotRow != i) {
      std::swap(A.data[i], A.data[pivotRow]);
      std::swap(I.data[i], I.data[pivotRow]);
    }

    // normalise pivot row
    pivot = A.data[i][i];
    for (int j = 0; j < n; ++j) {
      A.data[i][j] /= pivot;
      I.data[i][j] /= pivot;
    }

    // eliminate other rows
    for (int row = 0; row < n; ++row) {
      if (row == i) continue;
      double factor = A.data[row][i];
      for (int j = 0; j < n; ++j) {
        A.data[row][j] -= factor * A.data[i][j];
        I.data[row][j] -= factor * I.data[i][j];
      }
    }
  }

  // eeturn result as a new matrix
  matrix result(n, n);
  result = I;
  return result;
}

void matrix::print_matrix() {
  vector<int> dim = this->get_dimension_int();

  for (size_t i = 0; i < dim[0]; i++) {
    for (size_t j = 0; j < dim[1]; j++) {
      cout << this->data[i][j] << " ";
    }
    cout << endl;
  }
}

void matrix::insert_num(int a, int b, int num) { this->data[a][b] = num; }

void matrix::insert_matrix(std::vector<std::vector<double>> &mat) {
  this->data = mat;
}

matrix matrix::transpose() const {
  vector<int> dim = this->get_dimension_int();
  matrix trans(dim[1], dim[0]);

  for (size_t i = 0; i < dim[0]; i++) {
    for (size_t j = 0; j < dim[1]; j++) {
      trans.data[j][i] = this->data[i][j];
    }
  }

  return trans;
}

void matrix::scalar_multiply(double num) {
  vector<int> dim = this->get_dimension_int();

  for (size_t i = 0; i < dim[0]; i++) {
    for (size_t j = 0; j < dim[1]; j++) {
      this->data[i][j] *= num;
    }
  }
}

matrix matrix::hadamard(const matrix &other) const {
  vector<int> this_dim = this->get_dimension_int();
  vector<int> other_dim = this->get_dimension_int();
  if (this_dim != other_dim) {
    cout << "hadamard dimensions don't match";
    return matrix();
  }

  matrix res(this_dim[0], this_dim[1]);

  for (size_t i = 0; i < this_dim[0]; i++) {
    for (size_t j = 0; j < this_dim[1]; j++) {
      res.data[i][j] = this->data[i][j] * other.data[i][j];
    }
  }
  return res;
}

matrix matrix::apply_function(const std::function<double(double)> &func) const {
  std::vector<int> dim = this->get_dimension_int();
  matrix result(dim[0], dim[1]);

  for (int i = 0; i < dim[0]; ++i) {
    for (int j = 0; j < dim[1]; ++j) {
      result.data[i][j] = func(this->data[i][j]);
    }
  }

  return result;
}

void matrix::randomise(double min, double max) {
  random_device rd;
  mt19937 gen(rd());
  uniform_real_distribution<> dis(min, max);

  vector<int> dim = this->get_dimension_int();

  for (size_t i = 0; i < dim[0]; i++) {
    for (size_t j = 0; j < dim[1]; j++) {
      this->data[i][j] = dis(gen);
    }
  }
}

matrix matrix::from_vector(const std::vector<double> &vec) {
  matrix m(vec.size(), 1);
  for (size_t i = 0; i < vec.size(); i++) {
    m.data[i][0] = vec[i];
  }
  return m;
}

std::vector<double> matrix::to_vector() const {
  std::vector<double> vec;
  std::vector<int> dim = this->get_dimension_int();

  if (dim[1] != 1) {
    std::cerr << "Matrix is not a column vector.\n";
    return vec;
  }

  for (int i = 0; i < dim[0]; ++i) {
    vec.push_back(this->data[i][0]);
  }

  return vec;
}

// Returns a 1 x numCols matrix where each element is the sum of the column
// elements
matrix matrix::sum_rows() const {
  vector<int> dim = this->get_dimension_int();

  int rows = dim[0];
  int cols = dim[1];
  matrix result(1, cols);

  for (int j = 0; j < cols; ++j) {
    double colSum = 0;
    for (int i = 0; i < rows; ++i) {
      colSum += this->data[i][j];
    }
    result.data[0][j] = colSum;
  }
  return result;
}