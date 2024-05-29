function H = proj_defpos(M)

% Projection on the space of positive-definite matrix

% Input:    
%       - M: square symmetric matrix

% Output:
%       - H: square symmetric and positive-definite matrix obtained by
%       setting the non-negative eigenvalues to 0.01;

[P,K] = eig(M); K = diag(K);
K = subplus(K)+0.1; K = diag(K);
H = P*K*P';