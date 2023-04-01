function [logL, Rt, likelihoods, Qt]=dcc_mvgarch_Clikelihood(parameter,stdresid,data)

% Second-step composite-likelihood estimation of the scalar DCC process
% Contiguous overlapping pairs are considered, following the 2MSCLE method
% of Pakel, Shephard, Sheppard and Engle (2021), 'Fitting vast dimensional 
% time varying covariance models', Journal of Business & Economic
% Statistics

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
a=parameter(1);
b=parameter(2);

Qbar=cov(stdresid);
Qt=zeros(N,N,T);
Rt=zeros(N,N,T);
Qt(:,:,1)=Qbar; Rt(:,:,1) = corr(data);
logL=0;
likelihoods=zeros(1,T); dim = N-1;
for t=2:T
    Qt(:,:,t)=Qbar*(1-a-b) + a*(stdresid(t-1,:)'*stdresid(t-1,:)) + b*Qt(:,:,t-1);
    Rt(:,:,t)=Qt(:,:,t)./(sqrt(diag(Qt(:,:,t)))*sqrt(diag(Qt(:,:,t)))');
    CL=0;
    for ii=1:N-1
        index1 = ii; index2 = ii+1;
        R_temp = [1,Rt(index1,index2,t);Rt(index1,index2,t),1]; obs = [stdresid(t,index1),stdresid(t,index2)];
        CL = CL+log(det(R_temp))+obs*inv(R_temp)*obs';
        
    end
    likelihoods(t)=CL/dim;
    logL=logL+likelihoods(t);
end
logL=(1/2)*logL;
likelihoods=(1/2)*likelihoods;