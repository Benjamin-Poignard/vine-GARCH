% Real data analysis: MSCI portfolio
clear
clc
% load the MSCI country stock indexes: 23 assets
Table = readtable('MSCI.xls');

data_MSCI = Table{1:end,[2:end]};
% transform into log-returns
Y = log(data_MSCI(2:end,:))-log(data_MSCI(1:end-1,:));

reaus = Y(:,1); ref = Y(:,7); reg = Y(:,8); reuk = Y(:,22);
reus = Y(:,23); rejap = Y(:,13); reit = Y(:,12); regb = Y(:,22);
resp = Y(:,19); rehk = Y(:,10); resg = Y(:,18); reaut = Y(:,2);
regr = Y(:,9); rebl = Y(:,3); renl = Y(:,14); refin = Y(:,6);

[T,N] = size(Y); T = T-1;
X = zeros(T*N,2*N); y = zeros(T*N,N);
% Withdraw the conditional-mean effect (VAR assumption)
for ii = 1:N
    X((ii-1)*T+1:ii*T,(ii-1)*2+1:ii*2) = [ones(T,1),Y(1:end-1,ii)];
    y((ii-1)*T+1:ii*T,ii) = Y(2:end,ii);
end
b = (X'*X)\(X'*y);
res_blk = (y' - b'*X')';
res = zeros(T,N);
for k = 1:N
    res(:,k) = res_blk((k-1)*T+1:k*T,k);
end

N = size(res,2);

% Selection of the central nodes in each tree: US, Germany, Japan
node = [23 8 13];
data = res; y = zeros(size(data,1),size(data,2));
for ii = 1:3
    y(:,ii) = data(:,node(ii));
    data(:,node(ii)) = [];
end
y(:,ii+1:end) = data; res = y;

% In-sample estimation period
T_in = 2500; T_out = T_in+1;
% The variance models are estimated using res_in
% The DM test is performed using res_out
returns_in = res(1:T_in,:); returns_out = res(T_out:end,:);

method = 'truncation';
% estimation up to 3rd level; partial correlation processes not estimated
% for tree T_4, ...., T_{N-1}
level = 2;

% Parametric assumption on the innovation process
[Correlation_parametric,~,parameters_vine_parametric,~] = dynamic_vine(returns_in,method,level);

% scalar DCC model
[parameters_dcc,Rt,H_in]=dcc_mvgarch(returns_in,'full');

% Out-of-sample forecasts for the scalar DCC based
% univariate GARCH(1,1) processes
h_oos=zeros(size(returns_out,1),size(returns_out,2));
index = 1;
for jj=1:size(returns_out,2)
    univariateparameters=parameters_dcc(index:index+1+1);
    [simulatedata, h_oos(:,jj)] = dcc_univariate_simulate(univariateparameters,1,1,returns_out(:,jj));
    index=index+1+1+1;
end
%clear jj
h_oos = sqrt(h_oos);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% DCC out-of-sample variance-covariance %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% scalar DCC out-of-sample correlation process
[~,Rt_dcc_oos,~,~]=dcc_mvgarch_process_oos(parameters_dcc,returns_out,returns_in,H_in);

Hdcc = zeros(N,N,size(returns_out,1));

for t = 1:size(returns_out,1)
    Hdcc(:,:,t) = diag(h_oos(t,:))*Rt_dcc_oos(:,:,t)*diag(h_oos(t,:));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% C-Vine out-of-sample variance-covariance %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% C-Vine parametric out-of-sample correlation process
[ ~,Rt_vine_parametric ] = Cvine_correlation_process_oos(parameters_vine_parametric,res,T_in,h_oos,returns_out,method,level);

% C-Vine parametric and non-parametric out-of-sample variance-covariance process
Hvine_parametric = zeros(N,N,size(returns_out,1)); 
T = size(returns_out,1);
for t = 1:T
    Hvine_parametric(:,:,t) = diag(h_oos(t,:))*Rt_vine_parametric(:,:,t)*diag(h_oos(t,:));
end


% Compute the GMVP weights for the scalar DCC, C-Vine
% parametric model over the out-of-sample period
wdcc = zeros(N,T); wvine_parametric = zeros(N,T); 
for t = 1:T
    wdcc(:,t)= GMVP(Hdcc(:,:,t));
    wvine_parametric(:,t)= GMVP(Hvine_parametric(:,:,t));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%% Average Oracle Model %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Average oracle estimator: Sigma_ao
train = 500; test = 500; B = 10000;
Sigma_ao = average_oracle(returns_in,train,test,B);
% Compute the GMVP portfolio: one weight only since one variance covariance
% matrix, estimated in-sample, is considered for the out-of-sample period
w_ao = GMVP(Sigma_ao);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% Sample Variance-Covariance %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute the GMVP portfolio for the sample variance covariance matrix,
% estimated in-sample
w_sample = GMVP(cov(returns_in));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%% Portfolio returns (out-of-sample) %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
e1 = zeros(T,1); e2 = zeros(T,1); e3 = zeros(T,1);
e4 = zeros(T,1); e5 = zeros(T,1); 
for t = 1:T
    e1(t) = wdcc(:,t)'*returns_out(t,:)'; % scalar DCC
    e2(t) = wvine_parametric(:,t)'*returns_out(t,:)'; % C-Vine paramtric
    e3(t) = w_ao'*returns_out(t,:)'; % Average oracle
    e4(t) = w_sample'*returns_out(t,:)'; % Sample variance-covariance
    e5(t) = sum(returns_out(t,:))/N; % Equi-distributed strategy
end

% Compute the out-of-sample mean portfolio return (+ annualization: *252 since daily returns)
return_port = [mean(e1) mean(e2) mean(e3) mean(e4) mean(e5)]*252;
% Compute the out-of-sample standard deviation for the portfolio return (+ annualization: *sqrt(252) since daily returns)
std_port = [std(e1) std(e2) std(e3) std(e4) std(e5)]*sqrt(252);

e1 = e1.^2; e2 = e2.^2; e3 = e3.^2; e4 = e4.^2; e5 = e5.^2; 
E = [e1 e2 e3 e4 e5];
% Generate the table for the DM test results
DM2 = [];
for kk = 1:size(E,2)
    DM = [];
    for ii = 1:size(E,2)
        DM = [ DM , dmtest(E(:,kk),E(:,ii),1) ];
    end
    DM2 = [DM2;DM];
end

% Model Confidence Test GMVP
[includedR, pvalsR_gmvp, excluded] = mcs(E,0.1,10000,12);
excl_select_model_gmvp = [excluded ;includedR];
[excl_select_model_gmvp pvalsR_gmvp]