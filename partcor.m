function pcor = partcor(C,given,j,k)

S11 = C(given,given); jk = [j,k];
S12 = C(given,jk); S21 = C(jk,given); %S22 = S(jk,jk);
if (length(given)>1)
    tem = linsolve(S11,S12);
    Om212 = S21*tem;
else
    tem = S12/S11; Om212 = OuterProduct(S21,tem);
end
om11 = 1-Om212(1,1); om22 = 1-Om212(2,2);
pcor = (C(j,k)-Om212(1,2))/sqrt(om11*om22);