/// The Perceptron, base part for neural networks. Represents (or tries to)
/// an artificial neuron. Term coined by Frank Rosenblatt (1957).
class Perceptron {
  Perceptron(int nInputs, {double? learningRate}) {
    this.weights = List<double>.filled(nInputs, 0.0);
    this.learningRate = learningRate ?? this.learningRate;
  }

  late final List<double> weights; // One per input, for the weighted sum.
  double bias = 0.0; // The "intersection" for the linear function.
  double learningRate = 0.01; // The rate of correction for the weights and
  // bias in each training iteration.

  /// The forward pass. The linear function, but with a weighted sum of the
  /// inputs plus the bias, passed through an activation function.
  double predict(List<double> inputs) {
    double z = bias;
    for (var i = 0; i < inputs.length; i++) z += inputs[i] * weights[i];
    return z >= 0 ? 1 : 0; // In this case is the "step" function.
  }

  /// Training loop over the training data.
  /// `X` are the inputs (as vectors). `y` are the expected outputs. 
  /// `epochs` are the number of iterations for the training loop (trying to 
  /// find a sweet spot between this value and the learning rate to spend the
  /// less compute and adjust the model the best).
  void train(List<List<double>> X, List<double> y, {int epochs = 1000}) {
    for (int i = 0; i <= epochs; i++) { // Training loop.
      double errors = 0;
      for (var i = 0; i < X.length; i++) {
        List<double> inputs = X[i];
        double prediction = predict(inputs); // Forward pass.
        double error = y[i] - prediction;
        if (error != 0) {
          errors += 1;
          for (var i = 0; i < this.weights.length; i++) {
            // Weight correction.
            this.weights[i] += this.learningRate * error * inputs[i]; 
          }
          this.bias += this.learningRate * error;  // Bias correction.
        }
      }
      if (errors == 0) {
        print("Converged at epoch ${i + 1}.");
        return;
      }
    }
    // If it reaches the epoch limit before having 0 errors.
    print("Did not converge after $epochs epochs.");
  }
}

void main() {
  List<List<double>> and_or_x_data = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
  ];

  List<double> and_y_data = [0, 0, 0, 1];
  List<double> or_y_data = [0, 1, 1, 1];

  List<List<double>> not_x_data = [
    [0],
    [1],
  ];
  List<double> not_y_data = [1, 0];

  print("=== AND Gate ===");
  final Perceptron p_and = Perceptron(2);
  p_and.train(and_or_x_data, and_y_data);
  for (var inputs in and_or_x_data) {
    print("${inputs} -> ${p_and.predict(inputs)}");
  }

  print("=== OR Gate ===");
  final Perceptron p_or = Perceptron(2);
  p_or.train(and_or_x_data, or_y_data);
  for (var inputs in and_or_x_data) {
    print("${inputs} -> ${p_or.predict(inputs)}");
  }

  print("=== NOT Gate ===");
  final Perceptron p_not = Perceptron(1);
  p_not.train(not_x_data, not_y_data);
  for (var inputs in not_x_data) {
    print("${inputs} -> ${p_not.predict(inputs)}");
  }

  print("=== NAND Gate (Stacked AND AND NOT NEURONS) ===");
  for (var inputs in and_or_x_data) {
    print("${inputs} -> ${p_not.predict([p_and.predict(inputs)])}");
  }

  print("=== NAND Gate (Single Perceptron) ===");
  List<double> nand_y_data = [1, 1, 1, 0];
  final Perceptron p_nand = Perceptron(2);
  p_nand.train(and_or_x_data, nand_y_data);
  for (var inputs in and_or_x_data) {
    print("${inputs} -> ${p_nand.predict(inputs)}");
  }

  List<double> xor_y_data = [0, 1, 1, 0];

  print("=== XOR Gate (single perceptron, fails) ===");
  final Perceptron p_xor = Perceptron(2);
  p_xor.train(and_or_x_data, xor_y_data);
  for (var i = 0; i < and_or_x_data.length; i++) {
    final double result = p_xor.predict(and_or_x_data[i]);
    final expected = xor_y_data[i];
    final String status = result == expected ? "OK" : "WRONG";
    print("${and_or_x_data[i]} -> ${result} (expected ${expected}) ${status}");
  }

  print("\n=== XOR Gate (multi-layer (stacked perceptrons) network. Stacked NAND) ===");
  for (var inputs in and_or_x_data) {
    print(
      "${inputs} -> ${p_and.predict([
        p_or.predict(inputs),
        p_not.predict([p_and.predict(inputs)]),
      ])}",
    );
  }

  print("\n=== XOR Gate (multi-layer network (stacked perceptrons). Single NAND) ===");
  for (var inputs in and_or_x_data) {
    print(
      "${inputs} -> ${p_and.predict([p_or.predict(inputs), p_nand.predict(inputs)])}",
    );
  }
}
