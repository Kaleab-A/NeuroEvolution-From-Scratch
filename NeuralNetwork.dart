import 'dart:math';
import 'Neuron.dart';
import 'Activation.dart';
import 'Error.dart';

// Input Example:
// Nodes -> [3, 5, 3] - 3 input nodes, 5 hidden nodes, 3 output nodes
// Hidden Node can be in any structure, there is no layers.
//  1  |  4 5  |  9
//  2  |   6   |  10
//  3  |  7 8  |  11
// Hidden Node: [[1, 2], [4, 6], [5, 6], [4, 7]] - like directed graph

// TODO Forgot to add bias
class NeuralNetwork {
  List<int> nodes;
  List<List<int>> connections = [];

  List<Neuron> inputNeurons = [];
  List<Neuron> outputNeurons = [];

  Function errorFx;
  List<Function> activationFunctions;

  double learningRate;

  NeuralNetwork(
      this.nodes, this.connections, this.activationFunctions, this.errorFx,
      [this.learningRate = 0.01]) {
    init(this.nodes);
  }

  void init(nodes) {
    List<Neuron> neurons = [];
    int cnt = 0;

    for (int i = 0; i < nodes.length; i++) {
      for (int j = 0; j < nodes[i]; j++) {
        if (i == 0) {
          neurons.add(Neuron(cnt, 0, true, false, 0, 0, [], [],
              (x, [derivative = false]) => x));
          inputNeurons.add(neurons[cnt]);
        } else if (i == nodes.length - 1) {
          print("$activationFunctions , $i");
          neurons.add(Neuron(
              cnt, 0, false, true, 0, 0, [], [], activationFunctions[i - 1]));
          outputNeurons.add(neurons[cnt]);
        } else {
          neurons.add(Neuron(
              cnt, 0, false, false, 0, 0, [], [], activationFunctions[i - 1]));
        }
        cnt += 1;
      }
    }

    // allConnection contains a dictionary with key is a nueronID and value is a list of 2 sets of forward and backward node ids
    Map<int, List<Set<int>>> allConnection = {};
    for (int i = 0; i < connections.length; i++) {
      int backwardNode = connections[i][0];
      int forwardNode = connections[i][1];
      if (allConnection.containsKey(backwardNode)) {
        allConnection[backwardNode]![0].add(forwardNode);
      } else {
        allConnection[backwardNode] = [
          Set<int>()..add(forwardNode),
          Set<int>()
        ];
      }
      if (allConnection.containsKey(forwardNode)) {
        allConnection[forwardNode]![1].add(backwardNode);
      } else {
        allConnection[forwardNode] = [
          Set<int>(),
          Set<int>()..add(backwardNode)
        ];
      }
    }

    for (int i = 0; i < neurons.length; i++) {
      int id = i;
      neurons[i].backwardNodes = allConnection[id]![1].length;
      neurons[i].forwardNodes = allConnection[id]![0].length;
      neurons[i].backwardConnections = [];
      neurons[i].forwardConnections = [];
      for (int j = 0; j < neurons.length; j++) {
        if (allConnection[id]![1].contains(j)) {
          neurons[i].backwardConnections.add(neurons[j]);
        }
        if (allConnection[id]![0].contains(j)) {
          neurons[i].forwardConnections.add(neurons[j]);
        }
      }
    }
  }

  List<double> forward(input) {
    for (int i = 0; i < inputNeurons.length; i++) {
      inputNeurons[i].value = input[i];
      inputNeurons[i].forward();
    }

    List<double> output = [];
    for (int i = 0; i < outputNeurons.length; i++) {
      output.add(outputNeurons[i].value);
    }

    return output;
  }

  List<double> calcError(List<double> expectedOutput) {
    List<double> error = [];
    for (int i = 0; i < outputNeurons.length; i++) {
      error.add(errorFx(outputNeurons[i].value, expectedOutput[i]));
    }
    return error;
  }

  void backward(List<double> expectedOutput) {
    List<double> errors = calcError(expectedOutput);
    double avgError = errors.reduce((a, b) => a + b) / errors.length;

    var outputVal = [];
    for (int i = 0; i < outputNeurons.length; i++) {
      outputVal.add(outputNeurons[i].value);
    }
    print(outputVal);
    print("Error: $errors, Average Error: $avgError");

    for (int i = 0; i < outputNeurons.length; i++) {
      outputNeurons[i].error = errors[i];
      outputNeurons[i].errorFxn = errorFx;
      outputNeurons[i].backward(expectedOutput[i], learningRate);
    }
  }

  void train(List<List<double>> trainX, List<List<double>> trainY, int epochs,
      int batchSize) {
    for (int i = 0; i < epochs; i++) {
      for (int j = 0; j < trainX.length; j += batchSize) {
        List<List<double>> batchX = trainX.sublist(j, j + batchSize);
        List<List<double>> batchY = trainY.sublist(j, j + batchSize);
        for (int k = 0; k < batchX.length; k++) {
          forward(batchX[k]);
          backward(batchY[k]);
        }
      }
    }
  }
}

// void main() {
//   var a = NeuralNetwork([
//     2,
//     5,
//     2
//   ], [
//     [0, 2],
//     [1, 2],
//     [1, 3],
//     [2, 4],
//     [3, 4],
//     [3, 5],
//     [4, 6],
//     [5, 7],
//     [5, 8],
//     [6, 7],
//     [6, 8]
//   ], [
//     Activation().relu,
//     Activation().relu,
//     Activation().relu,
//     Activation().linear
//   ], Error().meanSquareError);

//   a.train([
//     [1, 4]
//   ], [
//     [5, 0]
//   ], 20, 1);
// }

void main() {
  // Testing if simple perceptron can predict sum of 2 inputs
  var a = NeuralNetwork([
    2,
    1
  ], [
    [0, 2],
    [1, 2]
  ], [
    Activation().linear
  ], Error().meanSquareError);

  a.train([
    [1, 5]
  ], [
    [5]
  ], 5, 1);
}
