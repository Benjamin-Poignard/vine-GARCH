function PC = corr2pcorr(Correl,VineArray)

% corr2pcorr.m: performs the mapping from the classic correlation
% matrix to the partial correlation matrix, where the partial correlation
% structure, i.e., the sets of conditioning and conditioned variables are
% given by any arbitrary R-vine structure
% See Dißmann, Brechmann, Czado and Kurowicka (2013), 'Selecting and 
% estimating regular vine copulae and application to financial returns', 
% CSDA, 59, 52-69, for more details on vine array

% Inputs:
%        - Correl: d x d classic correlation matrix
%        - VineArray: lower triangular matrix/triangular array, with 
%        non-zero elements below (including) the main diogonal, providing
%        the vine structure

% Output:
%        - PC: d x d partial correlation matrix, where the entries are the
%        partial correlations for the edges represented in VineArray

d = size(Correl,2);
oldOrder = diag(VineArray); Correl = Correl(flip(oldOrder),flip(oldOrder));
% if d=2 there is nothing to compute
if (d==2)
    % when d=2, then no transformation required
    PCorr = Correl;
else
    % rotate towards notation in Kurowicka and Joe (2011), p. 9
    [VineArray,~] = reorderRVineMatrix(VineArray);
    A = rot90(VineArray,2);
    % initialize correlation matrix with correlation parameters of the model
    PCorr = eye(d);
    for j=2:d
        PCorr(1,j) = Correl(A(1,j),j);
        PCorr(j,1) = Correl(A(1,j),j);
    end
    % calculations for second tree
    for j=3:d
        a1 = A(1,j); a2 = A(2,j);
        PCorr(2,j) = (Correl(j,a2)-Correl(j,a1)*Correl(a1,a2))/sqrt((1-Correl(j,a1)^2)*(1-Correl(a1,a2)^2));
        PCorr(j,2) = PCorr(2,j);
    end
    % remaining trees
    if (d>3)
        for ell=3:d-1
            for j=ell+1:d
                given = A(1:(ell-1),j);
                PCorr(ell,j) = partcor(Correl,given,A(ell,j),j);
                PCorr(j,ell) = PCorr(ell,j);
            end
        end
    end
end
PC = rot90(PCorr,2);