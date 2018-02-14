#include "test.hh"

double getScalarValue(Tensor<double> tensor) {
  return ((double*)tensor.getStorage().getValues().getData())[0];
}

int main(int argc, char** argv) {

  int M = atoi(argv[1]);
  int nx = atoi(argv[2]);

  clock_t t1, t2, t3, t1cum(0), t2cum(0);

  random_device seed;
  default_random_engine rng(seed());
  uniform_real_distribution<double> dis01(0, 1);

  /*Tensor<double> result(0.0);
  Tensor<double> theta("theta", {M, M}, Format({Dense, Dense}));
  Tensor<double> z("z", {M}, Format({Dense}));
  Tensor<double> intermed("intermed", {M,M}, Format({Dense, Dense}));

  IndexVar i1, i2, i3, i4, i5;

  intermed(i1,i2) = z(i1)*theta(i1,i2);
  intermed.compile();

  result = z(i3)*intermed(i1,i2)*intermed(i2,i3)*theta(i1,i3);
  result.compile();*/

  vector<vector<double>> theta(M, vector<double>(M,0));
  vector<double> z(M,0);

  for (int n = 0; n < nx; n++) {

    t1 = clock();

    //theta.zero();
    //z.zero();
    //result.zero();

    for (int i = 0; i < M; i++) {
      for (int j = 0; j < i; j++) {
        double v = dis01(rng);
        //theta.insert({i,j}, v);
        //theta.insert({j,i}, v);
        theta[i][j] = v;
        theta[j][i] = v;
      }
      //z.insert({i}, dis01(rng));
      //theta.insert({i,i}, 0.);
      theta[i][i] = 0.;
      z[i] = dis01(rng);
    }

    //theta.pack();
    //z.pack();

    t2 = clock();

    /*intermed.assemble();
    intermed.compute();

    result.assemble();
    result.compute();*/

    double result(0);
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < M; j++) {
        for (int k = 0; k < M; k++) {
          result += z[i]*z[j]*z[k]*theta[i][j]*theta[i][k]*theta[j][k];
        }
      }
    }

    t3 = clock();

    t1cum += t3 - t1;
    t2cum += t3 - t2;

    //cerr << getScalarValue(result) << endl;
  }

  cout << "Time 1 average: " << ((float) t1cum)/CLOCKS_PER_SEC/nx << endl
       << "Time 2 average: " << ((float) t2cum)/CLOCKS_PER_SEC/nx << endl;

  return 0;
}