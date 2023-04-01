function [c,ceq] = dcc_constr(x,stdresid)

a = x(1); b = x(2);
Qbar=(stdresid'*stdresid).*(1/length(stdresid));
K=Qbar*(1-a-b);
E=eig(K);
k1 = -a+0.0001;
k2 = a-0.5;
k3 = 0.001-b;
k4 = b-0.999;
k5 = a+b-0.999;
c = [ -min(E) ; k1 ; k2 ; k3 ; k4 ; k5 ];
ceq = [];


