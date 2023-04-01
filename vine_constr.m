function [c_ineq,ceq] = vine_constr(x)

% parameter constraints for the C-vine GARCH processes located in tree
% level 2, 3, ....

% x: 3 x 1 vector of parameters

a = x(1); b = x(2); c = x(3);
c_ineq = [];
c_ineq = [ c_ineq ; abs(a) - 0.3 ];
c_ineq = [ c_ineq ; b - 0.999 ];
c_ineq = [ c_ineq ; 0.6 - b ];
c_ineq = [ c_ineq ; abs(c) - 0.4 ];
ceq = [];
