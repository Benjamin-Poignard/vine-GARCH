function [parameters,Rt,H]=dcc_mvgarch(data,method)

% Inputs:
%        - data: T x N vector of returns
%        - method: estimation method: 'full' or 'CL', with 'full' the full
%        the full likelihood-based estimation and 'CL' the composite
%        likelihood-based estimation
% Outputs:
%        - parameters: (Number of univariateparameters)*N+2 x 1 vector,
%        the two last ones are the scalar DCC parameters
%        - Rt: the correlation matrix process
%        - H: T x N matrix of the univariate conditional GARCH variances

[t,k]=size(data);
archP = 1; garchQ = 1;

optimoptions.Hessian = 'bfgs';
optimoptions.MaxRLPIter = 100000;
optimoptions.Algorithm = 'sqp';
optimoptions.TolCon = 1e-09;
optimoptions.TolRLPFun = 1e-09;
optimoptions.MaxSQPIter = 100000;
for i=1:k
    [univariate{i}.parameters, univariate{i}.likelihood, univariate{i}.stderrors, univariate{i}.robustSE, univariate{i}.ht, univariate{i}.scores] ...
        = fattailed_garch(data(:,i) , archP , garchQ , 'NORMAL',[], optimoptions);
    stdresid(:,i)=data(:,i)./sqrt(univariate{i}.ht);
end

optimoptions.Hessian = 'bfgs';
optimoptions.MaxRLPIter = 500000;
optimoptions.Algorithm = 'sqp';
optimoptions.TolCon = 1e-09;
optimoptions.TolRLPFun = 1e-09;
optimoptions.MaxSQPIter = 500000;
optimoptions.Diagnostics = 'on';
optimoptions.Jacobian = 'off';
optimoptions.Display = 'iter';

condi = true;
while condi
    dccstarting(1) = 0.0001+(0.03-0.001)*rand(1,1);
    dccstarting(2) = 0.8+(0.95-0.8)*rand(1,1);
    [c,~] = dcc_constr(dccstarting,stdresid);
    condi = any(c>0);
end

switch method
    case 'full'
        [dccparameters,~,~,~,~,~]=fmincon(@(x)dcc_mvgarch_likelihood(x,stdresid,data),dccstarting,[],[],[],[],[],[],@(x)dcc_constr(x,stdresid),optimoptions);
    case 'CL'
        [dccparameters,~,~,~,~,~]=fmincon(@(x)dcc_mvgarch_Clikelihood(x,stdresid,data),dccstarting,[],[],[],[],[],[],@(x)dcc_constr(x,stdresid),optimoptions);  
end
% Estimated parameters
parameters=[];
H=zeros(t,k);
for i=1:k
    parameters=[parameters;univariate{i}.parameters];
    H(:,i)=univariate{i}.ht;
end
parameters=[parameters;dccparameters'];

% Computation of the correlation matrix and likelihood
[~,Rt,~,~]=dcc_mvgarch_full_likelihood(parameters,data);
