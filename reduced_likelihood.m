function [L,l,Correlation] = reduced_likelihood(param,data,h)

% 'local' log-likelihood function of the C-vine GARCH process for the first
% C-vine tree

% Inputs: 
%        - param: 3 x 1 parameter vector of interest
%        - data: T x 2 matrix of observations for the variables 
%        corresponding to the conditioned set of the first level of the
%        vine tree
%        - h: T x 2 matrix of conditional GARCH(1,1) volatility processes
%        for the variables corresponding to the conditioned set of the
%        first level of the vine tree

% Outputs:
%        - L: 'local' log-likelihood function evaluated at param
%        - l: T x 1 log-likelihood function evaluated at param such that 
%        L = sum(l)/2
%        - Correlation: 2 x 2 x T dynamic correlation process generated
%        from the C-vine GARCH model


u = data./h; eta = u(:,1).*u(:,2);
% eta: T x 1 vector of innovations entering in the vine GARCH
% process, under the parametric assumption (conditionally Gaussian 
% variables)

[T,N] = size(data); M_vec = vech_on(corrcoef(data),N);
M = zeros(T-1,1); M(1) = M_vec; M_tan = zeros(T-1,1);
PCorrelation = zeros(N,N,T-1);
Correlation = zeros(N,N,T); Correlation(:,:,1) = vech_off(M_vec,N);

for t = 2:T
    M_tan(t) = param(1) + param(2).*tan((pi/2).*M(t-1)) + param(3).*eta(t-1);
    M(t) = (2/pi).*atan(M_tan(t));
    PCorrelation(:,:,t) = vech_off(M(t),N);
    Correlation(:,:,t) = partial2corr_Cvine(PCorrelation(:,:,t));
end
clear t
l = zeros(T,1); 
for t = 1:T
    l(t) = log(det(Correlation(:,:,t))) + u(t,:)*inv(Correlation(:,:,t))*u(t,:)';
end
L = sum(l)/2;
clear t