function [node,data] = select_Cvine_akt(data_input)

%   Selection of the central node according to an average
%   Kendall's tau measure (AKT): for each variable returns_k,
%   compute its AKT w.r.t. all other variables, where AKT is: 
%   sum_{1 \leq j \leq N, j \neq k}|\tau_{kj}| with \tau_{kj}
%   the Kendall's tau between returns_k and returns_j.
%   Then the variable with the highest AKT is selected as the
%   central node of the first tree; the central node of the
%   second tree is the variable with the second highest AKT.
%   All other central nodes for each vine tree are set
%   according to this criterion.

% Input:  
%       - data_input: T x N matrix of observations

% Outputs:
%       - node: (N-1) x 1 vector of ordered indices, where the first index 
%       is the central node of the first tree of the C-vine; the second 
%       index is the central node of the second tree of the C-vine; and the
%       like
%       - data: T x N re-ordered matrix of observations according to node,
%       where the first column is the variable that is the central node of
%       the first tree of the C-vine; and the like for the other columns

data = data_input;
N = size(data,2); tau = zeros(N,N);
for i = 1:N
    parfor j = 1:N
        tau(j,i) = corr(data(:,i),data(:,j),'type','Kendall');
    end
end
[~,node] = sort(sum(abs(tau)),'descend');
data = data_input(:,node);