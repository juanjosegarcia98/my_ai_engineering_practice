// Example data. Taken from the printer sales example dataset given in Chapter
// 13 - Linear regression and Correlation, from Lind, Marchal and Mason's
// textbook.

const List<double> xs = [
  20,
  40,
  20,
  30,
  10,
  10,
  20,
  20,
  20,
  30,
]; // Independent variable. Calls made by salesperson.
const List<double> ys = [30, 60, 40, 60, 30, 40, 40, 50, 30, 70]; 
// Dependent variable values. Printers sold by salesperson.

final n = xs.length; // Dataset length.