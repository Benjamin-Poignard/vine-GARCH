function P = corr2partial_Cvine(C)

% corr2partial_Cvine.m performs the mapping from the classic correlation
% matrix to the partial correlation matrix, where the partial correlation
% structure, i.e., the sets of conditioning and conditioned variables are
% given by the C-vine structure

% Input:
%        - C: d x d standard correlation matrix
%        The variable in the first column of the matrix of observations is
%        the central node of the first tree level, i.e., C(1,2),
%        C(1,3), ..., C(1,d) will be conditioned set of the edges in the
%        first tree level
%        The variable in the second column of the matrix of observations is
%        the central node of the second tree level, given variable 1, i.e.,
%        C(2,3), ...., C(2,d) will be conditioned set of the edges in the
%        second tree level, given variable 1

% Output:
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

d = size(C,2); P = eye(d,d);
for i = 2:d-1
    
    for k = i+1:d
        if (i == 2)
            P(i,k) = (C(i,k) - C(i-1,2)*C(i-1,k))/(sqrt((1-C(i-1,2)^2)*(1-C(i-1,k)^2)));
        else
            l = zeros(i-1,1);
            l(1) = (C(i,k) - C(1,i)*C(1,k))/(sqrt((1-C(1,i)^2)*(1-C(1,k)^2)));
            for n = 2:i-1
                l(n) = (l(n-1) - P(n,i)*P(n,k))/sqrt((1-P(n,i)^2)*(1-P(n,k)^2));
            end
            P(i,k) = l(end);
        end
        P(k,i) = P(i,k);
    end
end
P(1,2:end) = C(1,2:end); P(2:end,1) = P(1,2:end);