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

class NeuralNetwork {
  List<int> nodes;
  List<List<int>> connections = [];

  List<Neuron> inputNeurons = [];
  List<Neuron> outputNeurons = [];

  NeuralNetwork(this.nodes, this.connections) {
    init(this.nodes, this.connections);
  }

  void init(nodes, connections) {
    List<Neuron> neurons = [];
    int cnt = 0;

    for (int i = 0; i < nodes.length; i++) {
      for (int j = 0; j < nodes[i]; j++) {
        if (i == 0) {
          neurons.add(Neuron(cnt, 0, true, false, 0, 0, [], [], (x) => x));
          inputNeurons.add(neurons[cnt]);
        } else if (i == nodes.length - 1) {
          neurons.add(
              Neuron(cnt, 0, false, true, 0, 0, [], [], Activation().sigmoid));
          outputNeurons.add(neurons[cnt]);
        } else {
          neurons.add(
              Neuron(cnt, 0, false, false, 0, 0, [], [], Activation().relu));
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

  List<double> calcError(List<double> expectedOutput, Function errorFx) {
    List<double> error = [];
    for (int i = 0; i < outputNeurons.length; i++) {
      error.add(errorFx(outputNeurons[i].value, expectedOutput[i]));
    }
    return error;
  }
}

void main() {
  var a = NeuralNetwork([
    2,
    5,
    2
  ], [
    [0, 2],
    [1, 2],
    [1, 3],
    [2, 4],
    [3, 4],
    [3, 5],
    [4, 6],
    [5, 7],
    [5, 8],
    [6, 7],
    [6, 8]
  ]);
  print(a.forward([1, 2]));
}
