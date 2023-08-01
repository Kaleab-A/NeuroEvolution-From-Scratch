import 'dart:math';

List<double> randomInit(List<int> shape) {
  // Let's assume shape is always 1D
  var rng = Random();
  double temp = 0;

  List<double> result = [];

  for (int i = 0; i < shape[0]; i++) {
    temp = rng.nextDouble();
    if (rng.nextInt(2) == 0) {
      temp *= -1;
    }
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
  late List<double> forwardWeights = randomInit([forwardNodes]);
  Function activation;

  double value;

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

      value = activation(value);
    }

    if (!isOutput) {
      for (int i = 0; i < forwardConnections.length; i++) {
        forwardConnections[i].forward();
      }
    }

    print("$id, $value");

    return value;
  }

  void backward() {
    if (!isInput) {
      for (int i = 0; i < backwardConnections.length; i++) {
        backwardConnections[i].backward();
      }
    }
  }
}
