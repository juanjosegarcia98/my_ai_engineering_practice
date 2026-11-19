// 1 - Linear regression. Simple version using gradient descent.
import 'common.dart';

const int epochs = 35000; // Training loop iteration count.
const int printEach = 1000;
const double lr = 0.001782; // Learning rate.

void main() {
  double b = 0; // Slope of the linear function.
  double a = 0; // Intersection with the Y axis.

  for (int i = 0; i <= epochs; i++) { // Training loop.
    double db = 0; // Slope gradient.
    double da = 0; // Intersection gradient.
    double loss = 0; // Loss or cost value.

    for (var i = 0; i < n; i++) { // Iteration over each data pair.
      double x = xs[i];
      double actualY = ys[i];
      double predictedY = (x * b) + a; // Forward pass prediction.
      
      db += ((predictedY - actualY) * x) * 2/n; // Adding to the gradient for the
      // slope.
      da += (predictedY - actualY) * 2/n; // Adding to the gradient for the
      // intersection.
      loss += ((predictedY - actualY) * (predictedY - actualY)) / n; // Cost or
      // loss using Mean Square Error.
    }

    b -= lr * db; // Correction of the slope using the gradient and learning
    // rate.
    a -= lr * da; // Correction of the intersection using the gradient and
    // learning rate.

    if (i % printEach == 0) print("Epoch: $i | Loss: $loss | w: $b | b: $a");
  }
}
