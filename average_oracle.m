function Sigma_ao = average_oracle(data,train,test,B)

% Variance-covariance estimator based on the average oracle estimator of
% Bongiorno, Challet and Loeper (2023), 'Filtering time-dependent
% covariance matrices using time-independent eigenvalues',
% https://arxiv.org/pdf/2111.13109.pdf

% Inputs:
%        - data: T x N matrix of observations
%        - train: train interval size
%        - test: test interval size
%        - B: number of simulations

% Output:
%        - Sigma_ao: Variance-covariance estimator

[T,N] = size(data); Lambda_opt = zeros(N,B);
for ii = 1:B
    t_ii = randi([train+1 T-test-1],1);
    S_train = cov(data(t_ii-train:t_ii-1,:)); S_test_ii = cov(data(t_ii:t_ii+test,:));
    [V_train_ii,~] = eig(S_train); Lambda_opt(:,ii) = diag(V_train_ii'*S_test_ii*V_train_ii);
end
[V_train,~] = eig(cov(data));
Sigma_ao = V_train*diag(mean(Lambda_opt,2))*V_train';