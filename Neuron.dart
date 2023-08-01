import 'dart:math';

List<double> randomInit(List<int> shape) {
  // Let's assume shape is always 1D
  var rng = Random();
  double temp = 0;

  List<double> result = [];

  for (int i = 0; i < shape[0]; i++) {
    temp = rng.nextDouble();
    // if (rng.nextInt(2) == 0) {
    //   temp *= -1;
    // }
    result.add(temp);
  }

  return result;
}

class Neuron {
  int id;

  bool isInput;
  bool isOutput;

  int backwardNodes;
  int forwardNodes;

  List<Neuron> backwardConnections;
  late List<double> backwardWeights = randomInit([backwardNodes]);
  List<Neuron> forwardConnections;
  Function activation;

  double value;
  double valueBeforeActivation = 0;

  double error = 0;
  Function errorFxn = () => 0;

  Neuron(
      this.id,
      this.value,
      this.isInput,
      this.isOutput,
      this.backwardNodes,
      this.forwardNodes,
      this.backwardConnections,
      this.forwardConnections,
      this.activation);

  double forward() {
    if (!isInput) {
      value = 0;
      for (int i = 0; i < backwardConnections.length; i++) {
        value += backwardConnections[i].value * backwardWeights[i];
      }

      valueBeforeActivation = value;
      value = activation(value);
    }

    if (!isOutput) {
      for (int i = 0; i < forwardConnections.length; i++) {
        forwardConnections[i].forward();
      }
    }

    // if (isOutput) {
    //   print("$id $value");
    // }

    return value;
  }

  void backward(double expectedValue, double learningRate,
      [double prevDerivative = 1]) {
    double dEdh = 1 * prevDerivative;
    List<double> dEdw = [];
    List<double> dEdx = [];

    // Calculate derivative of error
    if (isOutput) {
      dEdh *= errorFxn(value, expectedValue, true);
    }
    dEdh *= activation(valueBeforeActivation, true);

    // Calculate derivative of error with respect to weights
    for (int i = 0; i < backwardWeights.length; i++) {
      dEdw.add(dEdh * backwardConnections[i].value * learningRate);
    }

    // Calculate derivative of error with respect to backward neurons - passed to them for backpropagation
    for (int i = 0; i < backwardWeights.length; i++) {
      dEdx.add(dEdh * backwardWeights[i]);
    }

    // Update weights
    for (int i = 0; i < backwardWeights.length; i++) {
      backwardWeights[i] -= dEdw[i];
    }

    // If not input, continue backpropagation backwards
    if (!isInput) {
      for (int i = 0; i < backwardConnections.length; i++) {
        backwardConnections[i].backward(expectedValue, learningRate, dEdx[i]);
      }
    }
  }
}
