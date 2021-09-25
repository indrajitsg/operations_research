% include "globals.mzn";

% Decision Variables
var 5.0..9.0: x1;
var 0.0..0.7: x2;

% Objective Function
solve minimize 9.82 * x1 * x2 + 2 * x1;

% Constraints
constraint 2500 / (3.141 * x1 * x2) - 500 <= 0;

% constraint 2500/(3.141 * x1 * x2) - 3.141^2 * ((x1 * x1) + (x2 * x2))/0.5882 <= 0;

constraint - x1 + 2 <= 0;

constraint x1 - 14 <= 0;

constraint - x2 + 0.2 <= 0;

constraint x2 - 0.8 <= 0;

