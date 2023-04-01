function [c_ineq,ceq] = vine_constr_tree1(x)

% Parameter constraints for the partial correlation processes located in
% the first level of the C-vine tree

% x: 3 x 1 vector of parameters

a = x(1); b = x(2); c = x(3);
c_ineq = [];
c_ineq = [ c_ineq ; abs(a) - 0.3];
c_ineq = [ c_ineq ; b - 0.999 ];
c_ineq = [ c_ineq ; 0.75 - b ];
c_ineq = [ c_ineq ; abs(c) - 0.4 ];
ceq = [];
