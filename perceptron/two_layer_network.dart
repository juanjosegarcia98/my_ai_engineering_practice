import 'dart:math';

/// To solve the XOR problem (data not linearly separable). It is necessary
/// to stack more than one layer of perceptrons (Minsky and Papert, 1968). 
/// These are the "hidden" layers. Each neuron (perceptron), hidden or output,
/// has its own vector of weights (one for each input) and its own bias.
class TwoLayerNetwork {
  TwoLayerNetwork({double learningRate = 0.1}) {
    final Random random = Random();
    this.weightsHidden = List<List<double>>.generate(
      2,
      (index) => List<double>.generate(2, (index) => random.nextDouble() - 1.0),
    );
    this.biasesHidden = List<double>.generate(
      2,
      (index) => random.nextDouble() - 1.0,
    );
    this.weightsOutput = List<double>.generate(
      2,
      (index) => random.nextDouble() - 1.0,
    );
    this.biasOutput = random.nextDouble() - 1.0;
    this.learningRate = learningRate;
  }

  late List<List<double>> weightsHidden;
  late List<double> biasesHidden;
  late List<double> weightsOutput;
  late double biasOutput;
  late double learningRate;
  // Saving the computed outputs for the training steps.
  late List<double> hiddenOutputs;
  late double output;

  /// The activation function: the Sigmoid or Logistic function. Squashes the
  /// result to a value within the range (0,1). Continuous. Its derivative
  /// is easy to compute: s'(x) = s(x)(1 - s(x)).
  double sigmoid(x) {
    max(-500.0, min(500.0, x)); // Limit to the range of x to avoid overflow
    // when computing the exponential.
    return 1.0 / (1.0 + exp(-x));
  }

  /// The Perceptron's prediction.
  double forward(List<double> inputs) {
    this.hiddenOutputs = [];
    for (var i = 0; i < 2; i++) { // Here since it has 2 layers by design it
    // is built using two loops for convenience. This is from the inputs to
    // the hidden layer.
      double z = biasesHidden[i];
      List<double> weights = weightsHidden[i];
      for (var i = 0; i < inputs.length; i++) {
        z += weights[i] * inputs[i];
      }
      this.hiddenOutputs.add(sigmoid(z));
    }
    double zOutput = biasOutput;
    for (var i = 0; i < this.hiddenOutputs.length; i++) { // And this loop
      // is from the hidden outputs to the output layer (one neuron).
      zOutput += this.hiddenOutputs[i] * weightsOutput[i];
    }
    this.output = sigmoid(zOutput);
    return this.output;
  }

  /// Training loop. Similar to the single Perceptron. This is Backpropagation:
  /// the use of the Chain Rule and the derivatives to correct the weights and
  /// biases of all the neurons backwards after computing how wrong is the
  /// model.
  void train(List<List<double>> X, List<double> y, {int epochs = 10000}) {
    for (int i = 0; i <= epochs; i++) {
      double totalError = 0; // Only to know whether the network improves and
      // by how much each time. It has no further use here.
      for (var i = 0; i < X.length; i++) {
        List<double> inputs = X[i];
        double output = forward(inputs);
        double error = y[i] - output; // Simple error.
        totalError += pow(error, 2);

        // Output gradient = error * s'(output).
        double dOutput = error * output * (1 - output);
        List<double> savedWOutput = List.from(this.weightsOutput);
        List<double> hiddenDeltas = [];

        for (var i = 0; i < 2; i++) { 
          // Loop for the hidden neurons gradients. 
          // DeltaOutput * OutputWeight[i] * s'(HiddenOutput[i]).
          double h = this.hiddenOutputs[i];
          double hd = dOutput * savedWOutput[i] * h * (1 - h);
          hiddenDeltas.add(hd);
        }

        for (var i = 0; i < 2; i++) {
          // Output weights correction.
          // LR * DeltaOutput * HiddenOutput[i].
          this.weightsOutput[i] +=
              this.learningRate * dOutput * this.hiddenOutputs[i];
        }
        // Output bias correction.
        this.biasOutput += this.learningRate * dOutput;

        for (var i = 0; i < 2; i++) { // Iteration over each hidden neuron.
          // Hidden weights correction.
          for (var j = 0; j < inputs.length; j++) {
            // Iteration over each input.
            this.weightsHidden[i][j] +=
                this.learningRate * hiddenDeltas[i] * inputs[j];
          }
          // Hidden biases correction.
          this.biasesHidden[i] += this.learningRate * hiddenDeltas[i];
        }
      }
      if (i % 100 == 0) print("Epoch: $i | Total error: $totalError");
    }
  }
}

void main() {
  final TwoLayerNetwork net = TwoLayerNetwork(learningRate: 2);
  List<List<double>> xor_x_data = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
  ];
  List<double> xor_y_data = [0, 1, 1, 0];

  net.train(xor_x_data, xor_y_data);

  print("\n=== XOR Gate (2-layer network, with backpropagation and sigmoid activation) ===");
  for (var i = 0; i < xor_x_data.length; i++) {
    final double result = net.forward(xor_x_data[i]);
    final int predicted = result >= .5 ? 1 : 0;
    print("${xor_x_data[i]} -> ${result} (rounded: ${predicted}, expected ${xor_y_data[i]})");
  }
}