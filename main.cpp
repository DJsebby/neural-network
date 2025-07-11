#include "matrix.h"

int main() {
  matrix a(3, 3);
  matrix b(3, 3);

  std::vector<std::vector<double>> mat = {{-8, 2, 3}, {-4, 5, 8}, {7, 10, 9}};
  a.insert_matrix(mat);
  mat = {{1, 2, 3}, {-4, 5, 8}, {3, 0, 9}};
  b.insert_matrix(mat);
  matrix trans = a.transpose();
  matrix inv = a.inverse();
  //   inv.print_matrix();
  a.print_matrix();
  matrix ad = a.subtract(b);
  ad.print_matrix();
  //   trans.print_matrix();
  return 0;
}