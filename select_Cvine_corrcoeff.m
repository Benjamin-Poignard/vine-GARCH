function [node,data] = select_Cvine_corrcoeff(data_input)

%   Selection of the central node according to an averaged
%   sample linear correlation measure (ALC): for each variable returns_k,
%   compute its AC w.r.t. all other variables, where AC is: 
%   sum_{1 \leq j \leq N, j \neq k}|\corr_{kj}| with \corr_{kj}
%   the sample correlation between returns_k and returns_j.
%   Then the variable with the highest ALC is selected as the
%   central node of the first tree; the central node of the
%   second tree is the variable with the second highest AC.
%   All other central nodes for each vine tree are set
%   according to this criterion.

% Input:  
%       - data_input: T x N matrix of observations

% Outputs:
%       - node: 1 x (N-1) vector of ordered indices, where the first index 
%       is the central node of the first tree of the C-vine; the second 
%       index is the central node of the second tree of the C-vine; and the
%       like
%       - data: T x N re-ordered matrix of observations according to node,
%       where the first column is the variable that is the central node of
%       the first tree of the C-vine; and the like for the other columns

data = data_input;
N = size(data,2); tau = zeros(N,N);
for i = 1:N
    for j = 1:N
        tau(j,i) = corr(data(:,i),data(:,j));
    end
end
[~,node] = sort(sum(abs(tau)),'descend');
data = data_input(:,node);