function Mt = vechu_on(M,d)

% Inputs: 
% matrice M
% d: taille de la matrice

Mt = [];

for i = 1:(d-1)
    
    Mt = [ Mt ; M(1:i,i+1) ];
    
end

