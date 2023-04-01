function Correl = pcorr2corr(PCorr,VineArray)

% pcorr2corr.m: performs the mapping from the partial correlation
% matrix to the classic correlation matrix, where the partial correlation
% structure, i.e., the sets of conditioning and conditioned variables are
% given by any arbitrary R-vine structure

% See Dißmann, Brechmann, Czado and Kurowicka (2013), 'Selecting and
% estimating regular vine copulae and application to financial returns',
% CSDA, 59, 52-69, for more details on vine array

% Inputs:
%        - PCorr: d x d partial correlation matrix
%        - VineArray: lower triangular matrix/triangular array, with
%        non-zero elements below (including) the main diogonal, providing
%        the vine structure

% Output:
%        - Correl: d x d classic correlation matrix, where the entries are
%        the classic correlations for the edges represented in VineArray

d = size(PCorr,2);

% if d=2 there is nothing to compute
PCorr = rot90(PCorr,2);
if (d==2)
    % when d=2, then no transformation required
    Correl = PCorr;
else
    % rotate towards notation in Kurowicka and Joe (2011), p. 9
    [VineArray,oldOrder] = reorderRVineMatrix(VineArray);
    A = rot90(VineArray,2);
    % initialize correlation matrix with correlation parameters of the model
    Correl = eye(d);
    for j=2:d
        a1 = A(1,j);
        Correl(a1,j) = PCorr(1,j);
        Correl(j,a1) = PCorr(1,j);
    end
    % calculations for second tree
    for j=3:d
        a1 = A(1,j); a2 = A(2,j);
        Correl(j,a2) = Correl(j,a1)*Correl(a1,a2) + PCorr(2,j)*sqrt((1-Correl(j,a1)^2)*(1-Correl(a1,a2)^2));
        Correl(a2,j) = Correl(j,a2);
    end
    % remaining trees
    if (d>3)
        for ell=3:d-1
            for j=ell+1:d
                given = A(1:(ell-1),j);
                S11 = Correl(given,given);
                anew = A(ell,j);
                jk = [anew,j];
                S12 = Correl(given,jk); S21 = Correl(jk,given);
                tem = linsolve(S11,S12);
                Om212 = S21*tem;
                om11 = 1-Om212(1,1); om22 = 1-Om212(2,2);
                tem12 = PCorr(ell,j)*sqrt(om11*om22);
                Correl(anew,j) = tem12 + Om212(1,2);
                Correl(j,anew) = Correl(anew,j);
            end
        end
    end
end
[~,id] = sort(flip(oldOrder')); % Correl_gross = Correl;
Correl = Correl(id,id);