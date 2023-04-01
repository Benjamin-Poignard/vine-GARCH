function C = partial2corr_Cvine(P)

% partial2corr_Cvine.m performs the mapping from the partial correlation
% matrix to the classic correlation matrix, where the partial correlation
% structure, i.e., the sets of conditioning and conditioned variables are
% given by the C-vine structure

% Input: 
%        - P: d x d partial correlation matrix
%        Here, the edges of the first tree in the C-vine are represented in
%        P(1,2), P(1,3),..., and P(1,d): the variable in the first column
%        of the matrix of observations is the central node of the first
%        tree level
%        The edges in the second tree, given variable 1, that is (2,3|1),
%        ..., (2,d|1) are in P(2,3), P(2,4), ..., P(2,d): the variable in 
%        the second column of the matrix of observations is the central
%        node of the second tree level, given variable 1.
%        And the like for the remaining trees

% Output:
%        - C: d x d classic correlation matrix

d = size(P,2); C = eye(d,d);
for i = 2:d-1
    
    for k = i+1:d
        if (i == 2)
            C(i,k) = P(i,k)*sqrt((1-P(i-1,2)^2)*(1-P(i-1,k)^2)) + P(i-1,2)*P(i-1,k);
        else
            l = zeros(i-1,1);
            l(1) = P(i,k)*sqrt((1-P(i-1,i)^2)*(1-P(i-1,k)^2)) + P(i-1,i)*P(i-1,k);
            for n = 2:i-1
                l(n) = l(n-1)*sqrt((1-P(i-n,i)^2)*(1-P(i-n,k)^2)) + P(i-n,i)*P(i-n,k);
            end
            C(i,k) = l(end);
            clear l
        end
        C(k,i) = C(i,k);
    end
end
C(1,2:end) = P(1,2:end); C(2:end,1) = C(1,2:end);