function [c,ceq] = garch_constr(x)

const = x(1); a = x(2); b = x(3);
c = [ eps-const;
    eps-a;
    eps-b;
    a+b-0.99];
ceq = [];


