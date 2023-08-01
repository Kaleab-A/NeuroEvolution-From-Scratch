import 'dart:math';

class Error {
  // List<double> meanSquareError(List<double> output, List<double> expectedOutput,
  //     [bool derivative = false]) {
  //   List<double> error = [];

  //   if (!derivative) {
  //     for (int i = 0; i < output.length; i++) {
  //       error.add(
  //           (output[i] - expectedOutput[i]) * (output[i] - expectedOutput[i]));
  //     }
  //   } else {
  //     for (int i = 0; i < output.length; i++) {
  //       error.add(2 * (output[i] - expectedOutput[i]));
  //     }
  //   }
  //   return error;
  // }

  // List<double> crossEntropy(List<double> output, List<double> expectedOutput,
  //     [bool derivative = false]) {
  //   List<double> error = [];
  //   if (!derivative) {
  //     for (int i = 0; i < output.length; i++) {
  //       error.add(-expectedOutput[i] * log(output[i]) -
  //           (1 - expectedOutput[i]) * log(1 - output[i]));
  //     }
  //   } else {
  //     for (int i = 0; i < output.length; i++) {
  //       error.add((output[i] - expectedOutput[i]) /
  //           (output[i] * (1 - output[i]))); // TODO: Check if this is correct
  //     }
  //   }

  //   return error;
  // }

  double meanSquareError(double output, double expectedOutput,
      [bool derivative = false]) {
    if (!derivative) {
      return (output - expectedOutput) * (output - expectedOutput);
    } else {
      return 2 * (output - expectedOutput);
    }
  }

  double crossEntropy(double output, double expectedOutput,
      [bool derivative = false]) {
    if (!derivative) {
      return -expectedOutput * log(output) -
          (1 - expectedOutput) * log(1 - output);
    } else {
      return (output - expectedOutput) / (output * (1 - output));
    }
  }
}
