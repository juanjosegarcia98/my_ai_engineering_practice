// 1 - Linear regression. Simple straightforward version using the least
// squares method.
import 'common.dart';

void main() {
  final double sumX = xs.fold<double>(0.0, (a, b) => a + b);
  final double sumY = ys.fold<double>(0.0, (a, b) => a + b);
  final double sumSquaresX = xs
      .map((i) => i * i)
      .fold<double>(0.0, (a, b) => a + b);
  double sumXTimesY = 0;
  for (var i = 0; i < n; i++) sumXTimesY += xs[i] * ys[i];

  // Least squares method for calculating the slope b and intersection a for
  // the linear regression. Y' = bX + a.
  final double b =
      ((n * sumXTimesY) - (sumX * sumY)) / ((n * sumSquaresX) - (sumX * sumX));
  final double a = (sumY / n) - (b * (sumX / n));

  // Loss calculation using Mean Square Error.
  double loss = 0.0;
  for (var i = 0; i < n; i++) {
    double error = ((b * xs[i]) + a) - ys[i];
    loss += (error * error) / n;
  }

  print('Minimum squares method | Loss: $loss | b: $b | a: $a');
}
