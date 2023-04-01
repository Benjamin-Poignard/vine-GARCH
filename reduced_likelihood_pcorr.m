function [L,l,Correlation,PCorrelation] = reduced_likelihood_pcorr(param,level,data,h,correlp)

% 'local' log-likelihood function of the C-vine GARCH process for partial
% correlations located in trees 2, 3, 4, ..., N-1

% The variable ordering in data specified in dynamic_vine.m is of key:
% - the first column in data corresponds to the variable that is the
% central node in the first level of the C-vine tree
% - the second column in data corresponds to the variable that is the
% central node in the second level of the C-vine tree
% - and the like for the third, fourth, ..., (N-1)-th column

% Inputs: 
%        - param: 3 x 1 parameter vector of interest
%        - level: integer value corresponding to the level in the vine,
%        from tree 2, so level = 2, 3, ..., or N-1
%        - data: T x L matrix of observations, with L=level+1, where the 
%        L-2 first columns are the conditioning variables in 'level' of the
%        tree, the L-1 entry is the root variable in tree 'level', and the 
%        L entry forms the edge with the L-1 entry
%        - h: T x L matrix of GARCH(1,1) univariate processes of the
%        entries in the 'data' matrix
%        - correlp: vector containing all the partial correlation processes
%        located on the previous trees

% Outputs:
%        - L: 'local' log-likelihood function evaluated at param
%        - l: T x 1 log-likelihood function evaluated at param such that 
%        L = sum(l)/2
%        - Correlation: 2 x 2 x T dynamic correlation process generated
%        from the C-vine GARCH model
%        - PCorrelation: 2 x 2 x T dynamic partial correlation process 
%        generated from the C-vine GARCH model

[T,N] = size(data); M_vec = vech_on(corrcoef(data),N);
M_temp = zeros(T-1,1); K = vech_on(corr2partial_Cvine(vech_off(M_vec,N)),N);
M_tan = zeros(T-1,1);
Correlation = zeros(N,N,T); Correlation(:,:,1) = vech_off(M_vec,N);
PCorrelation = zeros(T,1); PCorrelation(1) = K(end);

temp = projection(data(1,:),h(1,:),Correlation(:,:,1));
eta = zeros(T-1,1); eta(1) = temp(end);

for t = 2:T
    M_tan(t) = param(1) + param(2).*tan((pi/2).*M_temp(t-1)) + param(3).*eta(t-1);
    M_temp(t) = (2/pi).*atan(M_tan(t));
    correlp(end,t) = M_temp(t); PCorrelation(t) = M_temp(t);
    Correlation(:,:,t) = partial2corr_Cvine(vech_off(correlp(:,t),N));
    temp = projection(data(t,:),h(t,:),Correlation(:,:,t));
    eta(t) = temp(end);
end
clear t

% Compute the 'local' log-likelihood function based on the pair of 
% variables (level,j), with j = level+1, ..., or N
u = data./h; l = zeros(T,1); u = [u(:,level),u(:,end)];
for t = 1:T
    l(t) = log(det(Correlation(end-1:end,end-1:end,t))) + u(t,:)*inv(Correlation(end-1:end,end-1:end,t))*u(t,:)';
end
L = sum(l)/2;
clear t