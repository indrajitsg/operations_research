# Nonlinear

var x1 >= 5, <=9;
var x2 >= 0.2, <=2;

minimize obj:
	9.82 * x1 * x2 + 2 * x1;

s.t. cons1:
	2500 / (3.141 * x1 * x2) - 500 <= 0;

s.t. cons2:
	2500/(3.141 * x1 * x2) - 3.141^2 * ((x1 * x1) + (x2 * x2))/0.5882 <= 0;

s.t. cons3:
	- x1 + 2 <= 0;

s.t. cons4:
	x1 - 14 <= 0;

s.t. cons5:
	- x2 + 0.2 <= 0;

s.t. cons6:
	x2 - 0.8 <= 0;

