# -------------------------------------------------------------------
# Non Linear Optimization
# Maximize f(x) = 2 * x0 * x1 + 2 * x0 - x0**2 - 2*x1**2
# subject to: 
#     x0**3 - x1 == 0
#     x1         >= 1
# -------------------------------------------------------------------
		
var x0 >= 0.0;
var x1 >= 0.0;

maximize obj:
	2*x0*x1 + 2*x0 - x0^2 - 2*x1^2;

s.t. c1:
	x0^3 - x1 == 0;

s.t. c2:
	x1 >= 1;

