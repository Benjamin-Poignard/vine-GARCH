function Mt = vech_on(M,d)

% Inputs: 
% matrice M
% d: taille de la matrice

Mt = [];

for i = 1:(d-1)
    
    Mt = [ Mt ; M(i+1:d,i) ];
    
end

