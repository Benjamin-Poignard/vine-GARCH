function [logL,Rt,likelihoods,Qt]=dcc_mvgarch_likelihood(params,stdresid,data)

% Full second-step log-likelihood of the scalar DCC process

% Inputs:
%        - parameter: 2 x 1 parameter vector of interest
%        - stdresid: T x N matrix of standardized residuals (data./volatility)
%        - data: T x N matrix of observations

% Outputs:
%        - logL: composite log-likelihood value evaluated at parameter
%        -  Rt: N x N x T correlation process generated from the scalar DCC
%        - likelihoods: T x 1 vector of log-likelihood evaluated at
%        parameter such that logL = sum(likelihoods)
%        - Qt: N x N x T process of the underlying Qt matrix generated from
%        the scalar DCC model

[T,N]=size(stdresid);
a=params(1);
b=params(2);

Qbar=cov(stdresid);
Qt=zeros(N,N,T);
Rt=zeros(N,N,T);
Qt(:,:,1)=Qbar; Rt(:,:,1) = corr(data);
logL=0;
likelihoods=zeros(1,T);
for t=2:T
    Qt(:,:,t)=Qbar*(1-a-b) + a*(stdresid(t-1,:)'*stdresid(t-1,:)) + b*Qt(:,:,t-1);
    Rt(:,:,t)=Qt(:,:,t)./(sqrt(diag(Qt(:,:,t)))*sqrt(diag(Qt(:,:,t)))');
    likelihoods(t)=log(det(Rt(:,:,t)))+stdresid(t,:)*inv(Rt(:,:,t))*stdresid(t,:)';
    logL=logL+likelihoods(t);
end
logL=(1/2)*logL;
likelihoods=(1/2)*likelihoods;