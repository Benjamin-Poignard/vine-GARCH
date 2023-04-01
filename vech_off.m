function M = vech_off(Mt,d)

% Inputs: 
% Mt vecteur de taille  d*(d-1)/2 x 1
% d: dimension du problème

M = tril(ones(d),-1);
M(M==1) = Mt;
M = M + M' + eye(d,d);

