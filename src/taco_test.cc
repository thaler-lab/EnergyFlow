#include "test.hh"

const uint M = 10;

int main() {

  random_device seed;
  default_random_engine rng(seed());
  uniform_real_distribution<double> dis01(0, 1);

  Tensor<double> result(0.0);
  Tensor<double> theta("theta", {M, M}, Format({Dense, Dense}));
  Tensor<double> z("z", {M}, Format({Dense}));

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < i; j++) {
      double v = dis01(rng);
      cout << v << endl;
      theta.insert({i,j}, i);
      theta.insert({j,i}, i);
    }
    z.insert({i}, i);
    theta.insert({i,i}, 0.0);
  }

  theta.pack();
  z.pack();

  IndexVar i1, i2, i3, i4, i5;

  result() = z(i1)*z(i2)*z(i3)*theta(i1,i2)*theta(i1,i3);

  result.evaluate();

  cout << result << endl;

  return 0;
}