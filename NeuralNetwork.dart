import 'dart:math';
import 'dart:io';
import 'Neuron.dart';
import 'Activation.dart';
import 'Error.dart';
import 'package:csv/csv.dart';

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
  List<double> errorTemp = [];

  bool fullyConnected = false;

  NeuralNetwork(
      {required this.nodes,
      required this.connections,
      required this.activationFunctions,
      required this.errorFx,
      required this.fullyConnected,
      required this.learningRate}) {
    init(this.nodes);
  }

  void init(nodes) {
    List<Neuron> neurons = [];
    int cnt = 0;

    // Creating Neurons (Input, Hidden, Output)
    for (int i = 0; i < nodes.length; i++) {
      for (int j = 0; j < nodes[i]; j++) {
        if (i == 0) {
          neurons.add(Neuron(cnt, 0, true, false, 0, 0, [], [],
              (x, [derivative = false]) => x));
          inputNeurons.add(neurons[cnt]);
        } else if (i == nodes.length - 1) {
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

    // If fully connected, create connections between all neurons
    if (fullyConnected) {
      List<List<int>> idRanges = [];
      int new_cnt = 0;

      for (int i = 0; i < nodes.length; i++) {
        idRanges.add([new_cnt, (new_cnt + nodes[i]) as int]);
        new_cnt += nodes[i] as int;
      }

      for (int i = 0; i < nodes.length - 1; i++) {
        for (int j = idRanges[i][0]; j < idRanges[i][1]; j++) {
          for (int k = idRanges[i + 1][0]; k < idRanges[i + 1][1]; k++) {
            connections.add([j, k]);
          }
        }
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

  void backward(List<double> expectedOutput, int x) {
    List<double> errors = calcError(expectedOutput);

    double avgError = errors.reduce((a, b) => a + b) / errors.length;
    errorTemp.add(avgError);

    for (int i = 0; i < outputNeurons.length; i++) {
      outputNeurons[i].error = errors[i];
      outputNeurons[i].errorFxn = errorFx;
      outputNeurons[i].backward(expectedOutput[i], learningRate);
    }
  }

  void train(
      {required List<List<double>> trainX,
      required List<List<double>> trainY,
      required int epochs,
      required int batchSize,
      required File file}) {
    file.writeAsStringSync("");
    for (int i = 0; i < epochs; i++) {
      for (int j = 0; j < trainX.length; j += batchSize) {
        List<List<double>> batchX = trainX.sublist(j, j + batchSize);
        List<List<double>> batchY = trainY.sublist(j, j + batchSize);
        for (int k = 0; k < batchX.length; k++) {
          forward(batchX[k]);
          backward(batchY[k], j);
        }
      }
      // Write the average of the last 4 values in errorTemp to a file
      double avgtemp = 0;
      for (int i = 0; i < 4; i++) {
        avgtemp += errorTemp[errorTemp.length - 1 - i];
      }
      avgtemp /= 4;
      file.writeAsStringSync(avgtemp.toString() + "\n", mode: FileMode.append);
    }
  }
}

File file1 = File("NN1.txt");
File file2 = File("NN2.txt");

// void main() {
//   List<List<double>> trainX = [
//     [0, 0],
//     [0, 1],
//     [1, 0],
//     [1, 1]
//   ];
//   List<List<double>> trainY = [
//     [0],
//     [1],
//     [1],
//     [1]
//   ];

//   // Testing if simple perceptron can predict sum of 2 inputs
//   var a = NeuralNetwork(
//       nodes: [
//         2,
//         2,
//         1
//       ],
//       connections: [],
//       activationFunctions: [
//         Activation().relu,
//         Activation().relu,
//       ],
//       errorFx: Error().meanSquareError,
//       fullyConnected: true,
//       learningRate: 0.01);

//   a.train(
//       trainX: trainX, trainY: trainX, epochs: 100, batchSize: 1, file: file1);
//   // -----------------------------------------------------------------------------------------------------------
//   var b = NeuralNetwork(
//       nodes: [
//         2,
//         16,
//         16,
//         1
//       ],
//       connections: [],
//       activationFunctions: [
//         Activation().relu,
//         Activation().relu,
//         Activation().relu,
//       ],
//       errorFx: Error().meanSquareError,
//       fullyConnected: true,
//       learningRate: 0.01);

//   b.train(
//       trainX: trainX, trainY: trainX, epochs: 100, batchSize: 1, file: file2);
// }

void main() {
  List<List<double>> trainX = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
  ];
  List<List<double>> trainY = [
    [0],
    [1],
    [1],
    [1]
  ];

  // Testing if simple perceptron can predict sum of 2 inputs
  var a = NeuralNetwork(
      nodes: [
        2,
        2,
        1
      ],
      connections: [],
      activationFunctions: [
        Activation().relu,
        Activation().relu,
      ],
      errorFx: Error().meanSquareError,
      fullyConnected: true,
      learningRate: 0.01);

  a.train(
      trainX: trainX, trainY: trainX, epochs: 100, batchSize: 1, file: file1);
  // -----------------------------------------------------------------------------------------------------------
  var b = NeuralNetwork(
      nodes: [
        2,
        16,
        16,
        1
      ],
      connections: [],
      activationFunctions: [
        Activation().relu,
        Activation().relu,
        Activation().relu,
      ],
      errorFx: Error().meanSquareError,
      fullyConnected: true,
      learningRate: 0.01);

  b.train(
      trainX: trainX, trainY: trainX, epochs: 100, batchSize: 1, file: file2);
}
