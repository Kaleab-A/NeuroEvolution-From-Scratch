import 'dart:math';

class Activation {
  double sigmoid(double x, [bool derivative = false]) {
    if (!derivative) {
      return 1 / (1 + exp(-x));
    } else {
      return sigmoid(x) * (1 - sigmoid(x));
    }
  }

  double relu(double x, [bool derivative = false]) {
    if (!derivative) {
      return max(0, x);
    } else {
      return x > 0 ? 1 : 0;
    }
  }

  double linear(double x, [bool derivative = false]) {
    if (!derivative) {
      return x;
    } else {
      return 1;
    }
  }

  // double tanh(double x) {
  //   return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
  // }

  // double leakyRelu(double x) {
  //   return max(0.01 * x, x);
  // }
}
