%% Several DGP for portfolio allocation analysis:

% DGP 1: one factor DGP
% DGP 2: two factors DGP
% DGP 3: sin/cos/constant dynamic correlation matrix
% DGP 4: autoregressive correlation process
% DGP 5: scalar DCC correlation process
% DGP 4: vine GARCH correlation process
%% On the selection of the C-vine

% The C-vine can be selected using the following three functions:
% - select_Cvine.m: the method is based on the average conditional
% kendall's tau non-parametric estimator. The method is time-consuming and
% may not be stable when the dimension of the data is larger then 5 or 6
% - select_Cvine_akt.m: selection of the central node according to an
% average Kendall's tau measure (AKT): for each variable returns_k,
% compute its AKT w.r.t. all other variables, where AKT is:
% sum_{1 \leq j \leq N, j \neq k}|\tau_{kj}| with \tau_{kj}
% the Kendall's tau between returns_k and returns_j. Then the variable with
% the highest AKT is selected as the central node of the first tree; the
% central node of the second tree is the variable with the second highest
% AKT. All other central nodes for each vine tree are set according to this
% criterion.
% - select_Cvine_corrcoeff.m: Selection of the central node according to an
% average sample linear correlation measure (ALC): for each variable
% returns_k, compute its AC w.r.t. all other variables, where AC is:
% sum_{1 \leq j \leq N, j \neq k}|\corr_{kj}| with \corr_{kj}
% the sample correlation between returns_k and returns_j. Then the variable
% with the highest AC is selected as the central node of the first tree;
% the central node of the second tree is the variable with the second
% highest AC. All other central nodes for each vine tree are set according
% to this criterion

% Before runing dynamic_vine.m on real/simulated data, the C-vine must be
% selected and the column indices of the data matrix of observations must
% be re-ordered accordingly (this is done by select_Cvine.m,
% select_Cvine_akt.m and select_Cvine_corrcoeff.m)


%% C-vine GARCH model estimation

% - dynamic_vine.m: main function to estimate the C-vine GARCH model, where
% the C-vine has been specified by the user or selected by select_Cvine.m,
% select_Cvine_akt.m or select_Cvine_corrcoeff.m
% If the user has specified the C-vine a priori, the ordering of the data
% matrix of observations is key to run the estimation algorithm: the first
% column will be the root of the first tree, the second column will be the
% root of the second tree given the variable in the first column, the third
% column will be the root in the third tree given the variables in the two
% first columns, etc.
% Full estimation or estimation with truncation can be performed; the
% estimation is performed edge-by-edge in each tree level, which allows to
% employ paraellel computation for each tree vine level: see Section 3.3 of
% Poignard and Fermanian (2019), 'Dynamic asset correlations based on
% vines', Econometric Theory, 35, 167-197
%
% - corr2partial_Cvine.m: performs the mapping from the classic correlation
% matrix to the partial correlation matrix, where the partial correlation
% structure, i.e., the sets of conditioning and conditioned variables are
% given by the C-vine structure, which is deduced from the ordering given
% in the columns of the data matrix of observations
% - partial2corr_Cvine: performs the mapping from the partial correlation
% matrix to the classic correlation matrix, where the partial correlation
% structure, i.e., the sets of conditioning and conditioned variables are
% given by the C-vine structure, which is deduced from the ordering given
% in the columns of the data matrix of observations
%
% - corr2pcorr.m: performs the mapping from the classic correlation
% matrix to the partial correlation matrix, where the partial correlation
% structure, i.e., the sets of conditioning and conditioned variables are
% given by any arbitrary R-vine structure.
% - pcorr2corr.m: performs the mapping from the partial correlation
% matrix to the classic correlation matrix, where the partial correlation
% structure, i.e., the sets of conditioning and conditioned variables are
% given by any arbitrary R-vine structure.
% Both corr2pcorr.m and pcorr2corr.m require the so-called vine array,
% which gives the vine structure. It is a lower triangular
% matrix/triangular array, with non-zero elements below (including) the
% main diogonal
% See the end of this script for examples
% See Dißmann, Brechmann, Czado and Kurowicka (2013), 'Selecting and
% estimating regular vine copulae and application to financial returns',
% CSDA, 59, 52-69, for more details on vine array
%
% On the parameter constraints for each partial correlation processes:
% the user may want to modify vine_constr_tree1.m and vine_contr.m, which
% provide the constraints on the Vine GARCH model. In particular, the
% upper/lower bounds on the constant 'a' and autoregressive parameter 'b'
% may be modified, depending on the dataset.
%% On the DCC model

% The code builds upon the MFE toolbox of K. Sheppard: see
% https://www.kevinsheppard.com/code/matlab/mfe-toolbox/

% The dcc model estimation only is considered (no standard error is
% computed)

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%% DGP 1: One factor correlation model %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear
clc
% Specify the desired dimension and full sample size
N = 10; T = 5001;

% Define the univariate GARCH(1,1) processes
hsim2 = zeros(T,N); hsim2(1,:) = 0.005.*ones(1,N);
hsim = zeros(T,N); hsim(1,:) = sqrt(hsim2(1,:));
% Define the variance-covariance and correlation matrices
Sigma = zeros(N,N,T); Correlation = zeros(N,N,T);
% Simulate the univariate GARCH(1,1) parameters (satisfying the
% stationarity constraints)
constant = 0.0001 + (0.009-0.0001)*rand(1,N);
[a_garch,b_garch] = simulate_garch_param(N);
% Define the T x N matrix of observations
returns = zeros(T,N);

% gamma: degree of freedom of the Student distribution
gamma = 3; % gamma > 2

% beta: coefficient entering in sin() (controls for the frequency)
beta = 600;
% N1: number of assets highly correlated with the factor
% N2: number of assets reasonably correlated with the factor
% N3: remaining of assets poorly correlated with the factor
N1 = 3; N2 = round(0.6*(N-N1)); N3 = N-N1-N2;
for t = 2:T
    
    hsim2(t,:) = constant + b_garch.*hsim2(t-1,:) + a_garch.*(returns(t-1,:).^2);
    hsim(t,:) = sqrt(hsim2(t,:));
    
    for ii = 1:N1
        a(ii) = 0.8-0.1*sin(2*pi*t/beta);
    end
    for ii = N1+1:N1+N2
        a(ii) = 0.4-0.3*sin(2*pi*t/beta);
    end
    for ii = N1+N2+1:N
        a(ii) = 0.5*sin(2*pi*t/beta);
    end
    for ii = 1:N
        for jj = 1:N
            if (ii==jj)
                Correlation(ii,jj,t) = 1;
            else
                Correlation(ii,jj,t) = a(ii)*a(jj);
            end
        end
    end
    clear a
    Sigma(:,:,t) = diag(hsim(t,:))*Correlation(:,:,t)*diag(hsim(t,:));
    % Verify whether the positive-definiteness condition is satisfied
    % If not, one may apply:
    % - Method 1: a transformation using (1-b) x St + b x Id, with S the
    % variance-covariance at time t, Id the identity matrix and b the
    % user-specified coefficient of linear combination
    % - Method 2: set the negative eigenvalues to 0.01
    if (min(eig(Sigma(:,:,t)))<eps)
        b = 0;
        while  (min(eig(Sigma(:,:,t)))<eps)
            b = b+0.001;
            Sigma(:,:,t) = (1-b)*Sigma(:,:,t)+b*eye(N);
        end
    end
    
    % Simulate the returns in the Student distribution with gamma degrees
    % of freedom. Alternatively, the returns can be simulated in a
    % multivariate normal distribution
    nu_temp = sqrt(gamma-2)*trnd(gamma,1,N)/sqrt(gamma);
    returns(t,:) = (Sigma(:,:,t)^(1/2)*nu_temp')';
    
end
% Discard the first matrix of Correlation and the first line of returns
Correlation = Correlation(:,:,2:end); returns = returns(2:end,:); T = length(returns);

%%%%%%%%%%%%%%%%%%%%%%%%%%% In-sample estimation %%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Vine-GARCH model %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%% C-Vine-GARCH is considered
%%%%%%%%%%%%%%  Step 1: select the central node according to AKT
[node,returns] = select_Cvine_akt(returns);

% Hereafter, work with the re-ordered matrix of observations 'returns' to
% estimate the C-Vine GARCH and DCC models

% Alternative: average conditional Kendall's tau
% [node,returns_new] = select_Cvine(returns)

% Alternative: average linear correlation coefficient
% [node,returns_new] = select_Cvine_corrcoeff(returns);

% Re-ordering of the indices in the true correlation matrix if the interest
% is to compare the true correlation processes with the estimated dcc and
% vine GARCH
for t = 1:T
    Correlation(:,:,t) = Correlation(node,node,t);
end

% Define the in-sample and out-of-sample periods
T_in = 4000; T_out = T_in+1;
returns_in = returns(1:T_in,:); returns_out = returns(T_out:end,:);

%%%%%%%%%%%%%% In-sample estimation
% level: the level up to which the estimation is performed
% level = 2: the partial correlation processes located in the 2 first
% levels of the C-vine tree are estimated; the remaining partial
% correlations are set as their sample partial correlations (computed from
% the sample correlation matrix and the underlying C-vine tree model)
level = 2; method = 'truncation';
[Rt_vine,Ht_vine,parameters_vine,~] = dynamic_vine(returns_in,method,level);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% scalar DCC model %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%% In-sample estimation
[parameters_dcc,Rt_dcc,H_in] = dcc_mvgarch(returns_in,'full');

%%%%%%%%%%%%%%%%%%%%%%%%% Out-of-sample evaluation %%%%%%%%%%%%%%%%%%%%%%%%
% Out-of-sample forecasts for the scalar DCC based
% univariate GARCH(1,1) processes
h_oos = zeros(size(returns_out,1),size(returns_out,2));
index = 1;
for jj=1:size(returns_out,2)
    univariateparameters = parameters_dcc(index:index+1+1);
    [simulatedata,h_oos(:,jj)] = dcc_univariate_simulate(univariateparameters,1,1,returns_out(:,jj));
    index=index+1+1+1;
end
% Generate the out-of-sample conditional GARCH(1,1) variances
h_oos = sqrt(h_oos); % transform into volatility

% scalar DCC out-of-sample correlation process
[~,Rt_dcc_oos,~,~] = dcc_mvgarch_process_oos(parameters_dcc,returns_out,returns_in,H_in);

% C-vine GARCH out-of-sample correlation process
[~,Rt_vine_oos] = Cvine_correlation_process_oos(parameters_vine,returns,T_in,h_oos,returns_out,method,level);

Hdcc = zeros(N,N,length(returns_out));
% scalar DCC out-of-sample variance covariance process
for t = 1:length(returns_out)
    Hdcc(:,:,t) = diag(h_oos(t,:))*Rt_dcc_oos(:,:,t)*diag(h_oos(t,:));
end

% truncated C-vine out-of-sample correlation process
Hvine = zeros(N,N,length(returns_out));
for t = 1:length(returns_out)
    Hvine(:,:,t) = diag(h_oos(t,:))*Rt_vine_oos(:,:,t)*diag(h_oos(t,:));
end

% Average oracle estimator
train = 500; test = 500; B = 10000;
Sigma_ao = average_oracle(returns_in,train,test,B);
w_ao = GMVP(Sigma_ao);

% Sample variance-covariance estimator
w_sample = GMVP(cov(returns_in));

% Computation of the out-of-sample averaged portfolio return and standard
% deviation
% First, obtain the GMVP based portfolio weights
wdcc = zeros(N,length(returns_out)); wvine = zeros(N,length(returns_out));
for t = 1:length(returns_out)
    wdcc(:,t)= GMVP(Hdcc(:,:,t));
    wvine(:,t)= GMVP(Hvine(:,:,t));
end

% Second, obtain the portfolio return series for each variance-covariance
% based model
e1 = zeros(length(returns_out),1); e2 = zeros(length(returns_out),1); e3 = zeros(length(returns_out),1);
e4 = zeros(length(returns_out),1); e5 = zeros(length(returns_out),1);
for t = 1:length(returns_out)
    e1(t) = wdcc(:,t)'*returns_out(t,:)';
    e2(t) = wvine(:,t)'*returns_out(t,:)';
    e3(t) = w_ao'*returns_out(t,:)';
    e4(t) = w_sample'*returns_out(t,:)';
    e5(t) = sum(returns_out(t,:))/N;
end

% Out-of-sample performance metrics: average portfolio return and standard
% deviation
average_return = 252*[mean(e1) mean(e2) mean(e3) mean(e4) mean(e5)];
sd_return = sqrt(252)*[std(e1) std(e2) std(e3) std(e4) std(e5)];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%% DGP 2: Two factors correlation model %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear
clc
% Specify the desired dimension and full sample size
N = 20; T = 5001;

% Define the univariate GARCH(1,1) processes
hsim2 = zeros(T,N); hsim2(1,:) = 0.005.*ones(1,N);
hsim = zeros(T,N); hsim(1,:) = sqrt(hsim2(1,:));
% Define the variance-covariance and correlation matrices
Sigma = zeros(N,N,T); Correlation = zeros(N,N,T);
% Simulate the univariate GARCH(1,1) parameters (satisfying the
% stationarity constraints)
constant = 0.0001 + (0.009-0.0001)*rand(1,N);
[a_garch,b_garch] = simulate_garch_param(N);
% Define the T x N matrix of observations
returns = zeros(T,N);

% gamma: degree of freedom of the Student distribution
gamma = 3;

% beta: coefficient entering in sin() (controls for the frequency)
beta1 = 600; beta2 = 800;
% N1: number of assets highly correlated with the factor
% N2: number of assets reasonably correlated with the factor
% N3: remaining of assets poorly correlated with the factor
N1 = 3; N2 = round(0.6*(N-N1)); N3 = N-N1-N2;
for t = 2:T
    
    hsim2(t,:) = constant + b_garch.*hsim2(t-1,:) + a_garch.*(returns(t-1,:).^2);
    hsim(t,:) = sqrt(hsim2(t,:));
    
    for ii = 1:N1
        a(ii) = (0.8-0.1*sin(2*pi*t/beta1))/sqrt(2);
        b(ii) = (0.8-0.1*sin(2*pi*t/beta2))/sqrt(2);
    end
    for ii = N1+1:N1+N2
        a(ii) = 0.4-0.3*sin(2*pi*t/beta1);
        b(ii) = 0.4-0.3*sin(2*pi*t/beta2);
    end
    for ii = N1+N2+1:N
        a(ii) = 0.5*sin(2*pi*t/beta1);
        b(ii) = 0.5*sin(2*pi*t/beta2);
    end
    for ii = 1:N
        for jj = 1:N
            if (ii==jj)
                Correlation(ii,jj,t) = 1;
            else
                Correlation(ii,jj,t) = a(ii)*a(jj)+b(ii)*b(jj);
            end
        end
    end
    clear a
    Sigma(:,:,t) = diag(hsim(t,:))*Correlation(:,:,t)*diag(hsim(t,:));
    
    % Verify whether the positive-definiteness condition is satisfied
    % If not, one may apply:
    % - Method 1: a transformation using (1-b) x St + b x Id, with S the
    % variance-covariance at time t, Id the identity matrix and b the
    % user-specified coefficient of linear combination
    % - Method 2: set the negative eigenvalues to 0.01
    if (min(eig(Sigma(:,:,t)))<eps)
        b = 0;
        while  (min(eig(Sigma(:,:,t)))<eps)
            b = b+0.001;
            Sigma(:,:,t) = (1-b)*Sigma(:,:,t)+b*eye(N);
        end
    end
    
    % Simulate the returns in the Student distribution with gamma degrees
    % of freedom. Alternatively, the returns can be simulated in a
    % multivariate normal distribution
    nu_temp = sqrt(gamma-2)*trnd(gamma,1,N)/sqrt(gamma);
    returns(t,:) = (Sigma(:,:,t)^(1/2)*nu_temp')';
    
end
% Discard the first matrix of Correlation and the first line of returns
Correlation = Correlation(:,:,2:end); returns = returns(2:end,:); T = length(returns);

%%%%%%%%%%%%%%%%%%%%%%%%%%% In-sample estimation %%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Vine-GARCH model %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%% C-Vine-GARCH is considered
%%%%%%%%%%%%%%  Step 1: select the central node according to AKT
[node,returns] = select_Cvine_akt(returns);

% Hereafter, work with the re-ordered matrix of observations 'returns' to
% estimate the C-Vine GARCH and DCC models

% Alternative: average conditional Kendall's tau
% [node,returns_new] = select_Cvine(returns)

% Alternative: average correlation coefficient
% [node,returns_new] = select_Cvine_corrcoeff(returns);

% Re-ordering of the indices in the true correlation matrix if the interest
% is to compare the true correlation processes with the estimated dcc and
% vine GARCH
for t = 1:T
    Correlation(:,:,t) = Correlation(node,node,t);
end

% Define the in-sample and out-of-sample periods
T_in = 4000; T_out = T_in+1;
returns_in = returns(1:T_in,:); returns_out = returns(T_out:end,:);

%%%%%%%%%%%%%% In-sample estimation
% level: the level up to which the estimation is performed
% level = 2: the partial correlation processes located in the 2 first
% levels of the C-vine tree are estimated; the remaining partial
% correlations are set as their sample partial correlations (computed from
% the sample correlation matrix and the underlying C-vine tree model)
level = 2; method = 'truncation';
[Rt_vine,Ht_vine,parameters_vine,~] = dynamic_vine(returns_in,method,level);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% scalar DCC model %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%% In-sample estimation
[parameters_dcc,Rt_dcc,H_in] = dcc_mvgarch(returns_in,'full');

%%%%%%%%%%%%%%%%%%%%%%%%% Out-of-sample evaluation %%%%%%%%%%%%%%%%%%%%%%%%
% Out-of-sample forecasts for the scalar DCC based
% univariate GARCH(1,1) processes
h_oos = zeros(size(returns_out,1),size(returns_out,2));
index = 1;
for jj=1:size(returns_out,2)
    univariateparameters = parameters_dcc(index:index+1+1);
    [simulatedata,h_oos(:,jj)] = dcc_univariate_simulate(univariateparameters,1,1,returns_out(:,jj));
    index=index+1+1+1;
end
% Generate the out-of-sample conditional GARCH(1,1) variances
h_oos = sqrt(h_oos); % transform into volatility

% scalar DCC out-of-sample correlation process
[~,Rt_dcc_oos,~,~] = dcc_mvgarch_process_oos(parameters_dcc,returns_out,returns_in,H_in);

% C-vine GARCH out-of-sample correlation process
[~,Rt_vine_oos] = Cvine_correlation_process_oos(parameters_vine,returns,T_in,h_oos,returns_out,method,level);

Hdcc = zeros(N,N,length(returns_out));
% scalar DCC out-of-sample variance covariance process
for t = 1:length(returns_out)
    Hdcc(:,:,t) = diag(h_oos(t,:))*Rt_dcc_oos(:,:,t)*diag(h_oos(t,:));
end

% truncated C-vine out-of-sample correlation process
Hvine = zeros(N,N,length(returns_out));
for t = 1:length(returns_out)
    Hvine(:,:,t) = diag(h_oos(t,:))*Rt_vine_oos(:,:,t)*diag(h_oos(t,:));
end

% Average oracle estimator
train = 500; test = 500; B = 10000;
Sigma_ao = average_oracle(returns_in,train,test,B);
w_ao = GMVP(Sigma_ao);

% Sample variance-covariance estimator
w_sample = GMVP(cov(returns_in));

% Computation of the out-of-sample averaged portfolio return and standard
% deviation
% First, obtain the GMVP based portfolio weights
wdcc = zeros(N,length(returns_out)); wvine = zeros(N,length(returns_out));
for t = 1:length(returns_out)
    wdcc(:,t)= GMVP(Hdcc(:,:,t));
    wvine(:,t)= GMVP(Hvine(:,:,t));
end

% Second, obtain the portfolio return series for each variance-covariance
% based model
e1 = zeros(length(returns_out),1); e2 = zeros(length(returns_out),1); e3 = zeros(length(returns_out),1);
e4 = zeros(length(returns_out),1); e5 = zeros(length(returns_out),1);
for t = 1:length(returns_out)
    e1(t) = wdcc(:,t)'*returns_out(t,:)';
    e2(t) = wvine(:,t)'*returns_out(t,:)';
    e3(t) = w_ao'*returns_out(t,:)';
    e4(t) = w_sample'*returns_out(t,:)';
    e5(t) = sum(returns_out(t,:))/N;
end

% Out-of-sample performance metrics: average portfolio return and standard
% deviation
average_return = 252*[mean(e1) mean(e2) mean(e3) mean(e4) mean(e5)];
sd_return = sqrt(252)*[std(e1) std(e2) std(e3) std(e4) std(e5)];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% DGP 3: sin/cos/constant correlation model %%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear
clc
% Specify the desired dimension and full sample size
N = 10; T = 5001;

% Define the univariate GARCH(1,1) processes
hsim2 = zeros(T,N); hsim2(1,:) = 0.005.*ones(1,N);
hsim = zeros(T,N); hsim(1,:) = sqrt(hsim2(1,:));
% Define the variance-covariance and correlation matrices
Sigma = zeros(N,N,T); Correlation = zeros(N,N,T);
% Simulate the univariate GARCH(1,1) parameters (satisfying the
% stationarity constraints)
constant = 0.0001 + (0.009-0.0001)*rand(1,N);
[a_garch,b_garch] = simulate_garch_param(N);
% Define the T x N matrix of observations
returns = zeros(T,N);

% gamma: degree of freedom of the Student distribution
gamma = 3;

normalisation = [100;200;500;1000;1200;1500;1800;2000];
a_coeff = 1+round(rand(N*(N-1)/2,1)*7);
b_select = 1+round(rand(N*(N-1)/2,1)*3); gg = {'cos','sin','const','mode'};
coefficient = -0.3+0.8*rand(N*(N-1)/2,2); d_const = randi([1 T],1);
L = zeros(N*(N-1)/2,T);

for t = 2:T
    
    hsim2(t,:) = constant + b_garch.*hsim2(t-1,:) + a_garch.*(returns(t-1,:).^2);
    hsim(t,:) = sqrt(hsim2(t,:));
    
    for ii = 1:N*(N-1)/2
        option = char(gg(b_select(ii)));
        switch option
            case 'cos'
                pp = cos(2*pi*t/normalisation(a_coeff(ii)));
            case 'sin'
                pp = sin(2*pi*t/normalisation(a_coeff(ii)));
            case 'mode'
                pp = mode(t/normalisation(a_coeff(ii)),1);
            case 'const'
                pp = double(t>d_const);
        end
        L(ii,t) = coefficient(ii,1)+coefficient(ii,2)*pp;
    end
    clear ii
    
    C = tril(vech_off(L(:,t),N)); Ctemp = C*C';
    Correlation(:,:,t) = Ctemp./(sqrt(diag(Ctemp))*sqrt(diag(Ctemp))');
    Sigma(:,:,t) = diag(hsim(t,:))*Correlation(:,:,t)*diag(hsim(t,:));
    % Verify whether the positive-definiteness condition is satisfied
    % If not, apply a transformation using (1-b) x St + b x Id, with S the
    % variance-covariance at time t, Id the identity matrix and b the
    % user-specified coefficient of linear combination
    if (min(eig(Sigma(:,:,t)))<eps)
        b = 0.01;
        Sigma(:,:,t) = (1-b)*Sigma(:,:,t)+b*eye(N);
    end
    nu_temp = sqrt(gamma-2)*trnd(gamma,1,N)/sqrt(gamma);
    returns(t,:) = (Sigma(:,:,t)^(1/2)*nu_temp')';
    
end
clear t
% Discard the first matrix of Correlation and the first line of returns
Correlation = Correlation(:,:,2:end); returns = returns(2:end,:); T = length(returns);
%%%%%%%%%%%%%%%%%%%%%%%%%%% In-sample estimation %%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Vine-GARCH model %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%% C-Vine-GARCH is considered
%%%%%%%%%%%%%%  Step 1: select the central node according to AKT
[node,returns] = select_Cvine_akt(returns);

% Hereafter, work with the re-ordered matrix of observations 'returns' to
% estimate the C-Vine GARCH and DCC models

% Alternative: average conditional Kendall's tau
% [node,returns_new] = select_Cvine(returns)

% Alternative: average correlation coefficient
% [node,returns_new] = select_Cvine_corrcoeff(returns);

% Re-ordering of the indices in the true correlation matrix if the interest
% is to compare the true correlation processes with the estimated dcc and
% vine GARCH
for t = 1:T
    Correlation(:,:,t) = Correlation(node,node,t);
end

% Define the in-sample and out-of-sample periods
T_in = 4000; T_out = T_in+1;
returns_in = returns(1:T_in,:); returns_out = returns(T_out:end,:);

%%%%%%%%%%%%%% In-sample estimation
% level: the level up to which the estimation is performed
% level = 2: the partial correlation processes located in the 2 first
% levels of the C-vine tree are estimated; the remaining partial
% correlations are set as their sample partial correlations (computed from
% the sample correlation matrix and the underlying C-vine tree model)
level = 2; method = 'truncation';
[Rt_vine,Ht_vine,parameters_vine,~] = dynamic_vine(returns_in,method,level);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% scalar DCC model %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%% In-sample estimation
[parameters_dcc,Rt_dcc,H_in] = dcc_mvgarch(returns_in,'full');

%%%%%%%%%%%%%%%%%%%%%%%%% Out-of-sample evaluation %%%%%%%%%%%%%%%%%%%%%%%%
% Out-of-sample forecasts for the scalar DCC based
% univariate GARCH(1,1) processes
h_oos = zeros(size(returns_out,1),size(returns_out,2));
index = 1;
for jj=1:size(returns_out,2)
    univariateparameters = parameters_dcc(index:index+1+1);
    [simulatedata,h_oos(:,jj)] = dcc_univariate_simulate(univariateparameters,1,1,returns_out(:,jj));
    index=index+1+1+1;
end
% Generate the out-of-sample conditional GARCH(1,1) variances
h_oos = sqrt(h_oos); % transform into volatility

% scalar DCC out-of-sample correlation process
[~,Rt_dcc_oos,~,~] = dcc_mvgarch_process_oos(parameters_dcc,returns_out,returns_in,H_in);

% C-vine GARCH out-of-sample correlation process
[~,Rt_vine_oos] = Cvine_correlation_process_oos(parameters_vine,returns,T_in,h_oos,returns_out,method,level);

Hdcc = zeros(N,N,length(returns_out));
% scalar DCC out-of-sample variance covariance process
for t = 1:length(returns_out)
    Hdcc(:,:,t) = diag(h_oos(t,:))*Rt_dcc_oos(:,:,t)*diag(h_oos(t,:));
end

% truncated C-vine out-of-sample correlation process
Hvine = zeros(N,N,length(returns_out));
for t = 1:length(returns_out)
    Hvine(:,:,t) = diag(h_oos(t,:))*Rt_vine_oos(:,:,t)*diag(h_oos(t,:));
end

% Average oracle estimator
train = 500; test = 500; B = 10000;
Sigma_ao = average_oracle(returns_in,train,test,B);
w_ao = GMVP(Sigma_ao);

% Sample variance-covariance estimator
w_sample = GMVP(cov(returns_in));

% Computation of the out-of-sample averaged portfolio return and standard
% deviation
% First, obtain the GMVP based portfolio weights
wdcc = zeros(N,length(returns_out)); wvine = zeros(N,length(returns_out));
for t = 1:length(returns_out)
    wdcc(:,t)= GMVP(Hdcc(:,:,t));
    wvine(:,t)= GMVP(Hvine(:,:,t));
end

% Second, obtain the portfolio return series for each variance-covariance
% based model
e1 = zeros(length(returns_out),1); e2 = zeros(length(returns_out),1); e3 = zeros(length(returns_out),1);
e4 = zeros(length(returns_out),1); e5 = zeros(length(returns_out),1);
for t = 1:length(returns_out)
    e1(t) = wdcc(:,t)'*returns_out(t,:)';
    e2(t) = wvine(:,t)'*returns_out(t,:)';
    e3(t) = w_ao'*returns_out(t,:)';
    e4(t) = w_sample'*returns_out(t,:)';
    e5(t) = sum(returns_out(t,:))/N;
end

% Out-of-sample performance metrics: average portfolio return and standard
% deviation
average_return = 252*[mean(e1) mean(e2) mean(e3) mean(e4) mean(e5)];
sd_return = sqrt(252)*[std(e1) std(e2) std(e3) std(e4) std(e5)];

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% DGP 4: Autoregressive correlation process %%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear
clc
% Specify the desired dimension and full sample size
N = 10; T = 5001;

% Define the univariate GARCH(1,1) processes
hsim2 = zeros(T,N); hsim2(1,:) = 0.005.*ones(1,N);
hsim = zeros(T,N); hsim(1,:) = sqrt(hsim2(1,:));
% Define the variance-covariance and correlation matrices
Sigma = zeros(N,N,T); Correlation = zeros(N,N,T);
% Simulate the univariate GARCH(1,1) parameters (satisfying the
% stationarity constraints)
constant = 0.0001 + (0.009-0.0001)*rand(1,N);
[a_garch,b_garch] = simulate_garch_param(N);
% Define the T x N matrix of observations
returns = zeros(T,N);

% gamma: degree of freedom of the Student distribution
gamma = 3;
[alpha,beta,zeta] = simulate_autocorrel_param(N);
dim = N*(N-1)/2;
rho = zeros(dim,T); rho_tan = zeros(dim,T);
eta = zeros(dim,1);
for t = 2:T
    
    hsim2(t,:) = constant + b_garch.*hsim2(t-1,:) + a_garch.*(returns(t-1,:).^2);
    hsim(t,:) = sqrt(hsim2(t,:));
    
    rho_tan(:,t) = alpha + beta.*tan((pi/2).*rho(:,t-1)) + zeta.*eta;
    rho(:,t) = (2/pi).*atan(rho_tan(:,t));
    Correlation(:,:,t) = vech_off(rho(:,t),N);
    
    Sigma(:,:,t) = diag(hsim(t,:))*Correlation(:,:,t)*diag(hsim(t,:));
    count = 0;
    % Verify whether the positive-definiteness condition is satisfied
    % If not, one may apply:
    % - Method 1: a transformation using (1-b) x St + b x Id, with S the
    % variance-covariance at time t, Id the identity matrix and b the
    % user-specified coefficient of linear combination
    % - Method 2: set the negative eigenvalues to 0.01
    if (min(eig(Sigma(:,:,t)))<eps)
        b = 0;
        while  (min(eig(Sigma(:,:,t)))<eps)
            b = b+0.001;
            Sigma(:,:,t) = (1-b)*Sigma(:,:,t)+b*eye(N);
        end
    end
    
    nu_temp = sqrt(gamma-2)*trnd(gamma,1,N)/sqrt(gamma);
    returns(t,:) = (Sigma(:,:,t)^(1/2)*nu_temp')';
    eta = [];
    for i = 1:N-1
        for j = i+1:N
            eta = [eta;(returns(t,i)/hsim(t,i)).*(returns(t,j)/hsim(t,j))];
        end
    end
    
end

clear t
% Discard the first matrix of Correlation and the first line of returns
Correlation = Correlation(:,:,2:end); returns = returns(2:end,:); T = length(returns);
%%%%%%%%%%%%%%%%%%%%%%%%%%% In-sample estimation %%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Vine-GARCH model %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%% C-Vine-GARCH is considered
%%%%%%%%%%%%%%  Step 1: select the central node according to AKT
[node,returns] = select_Cvine_akt(returns);

% Hereafter, work with the re-ordered matrix of observations 'returns' to
% estimate the C-Vine GARCH and DCC models

% Alternative: average conditional Kendall's tau
% [node,returns_new] = select_Cvine(returns)

% Alternative: average correlation coefficient
% [node,returns_new] = select_Cvine_corrcoeff(returns);

% Re-ordering of the indices in the true correlation matrix if the interest
% is to compare the true correlation processes with the estimated dcc and
% vine GARCH
for t = 1:T
    Correlation(:,:,t) = Correlation(node,node,t);
end

% Define the in-sample and out-of-sample periods
T_in = 4000; T_out = T_in+1;
returns_in = returns(1:T_in,:); returns_out = returns(T_out:end,:);

%%%%%%%%%%%%%% In-sample estimation
% level: the level up to which the estimation is performed
% level = 2: the partial correlation processes located in the 2 first
% levels of the C-vine tree are estimated; the remaining partial
% correlations are set as their sample partial correlations (computed from
% the sample correlation matrix and the underlying C-vine tree model)
level = 2; method = 'truncation';
[Rt_vine,Ht_vine,parameters_vine,~] = dynamic_vine(returns_in,method,level);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% scalar DCC model %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%% In-sample estimation
[parameters_dcc,Rt_dcc,H_in] = dcc_mvgarch(returns_in,'full');

%%%%%%%%%%%%%%%%%%%%%%%%% Out-of-sample evaluation %%%%%%%%%%%%%%%%%%%%%%%%
% Out-of-sample forecasts for the scalar DCC based
% univariate GARCH(1,1) processes
h_oos = zeros(size(returns_out,1),size(returns_out,2));
index = 1;
for jj=1:size(returns_out,2)
    univariateparameters = parameters_dcc(index:index+1+1);
    [simulatedata,h_oos(:,jj)] = dcc_univariate_simulate(univariateparameters,1,1,returns_out(:,jj));
    index=index+1+1+1;
end
% Generate the out-of-sample conditional GARCH(1,1) variances
h_oos = sqrt(h_oos); % transform into volatility

% scalar DCC out-of-sample correlation process
[~,Rt_dcc_oos,~,~] = dcc_mvgarch_process_oos(parameters_dcc,returns_out,returns_in,H_in);

% C-vine GARCH out-of-sample correlation process
[~,Rt_vine_oos] = Cvine_correlation_process_oos(parameters_vine,returns,T_in,h_oos,returns_out,method,level);

Hdcc = zeros(N,N,length(returns_out));
% scalar DCC out-of-sample variance covariance process
for t = 1:length(returns_out)
    Hdcc(:,:,t) = diag(h_oos(t,:))*Rt_dcc_oos(:,:,t)*diag(h_oos(t,:));
end

% truncated C-vine out-of-sample correlation process
Hvine = zeros(N,N,length(returns_out));
for t = 1:length(returns_out)
    Hvine(:,:,t) = diag(h_oos(t,:))*Rt_vine_oos(:,:,t)*diag(h_oos(t,:));
end

% Average oracle estimator
train = 500; test = 500; B = 10000;
Sigma_ao = average_oracle(returns_in,train,test,B);
w_ao = GMVP(Sigma_ao);

% Sample variance-covariance estimator
w_sample = GMVP(cov(returns_in));

% Computation of the out-of-sample averaged portfolio return and standard
% deviation
% First, obtain the GMVP based portfolio weights
wdcc = zeros(N,length(returns_out)); wvine = zeros(N,length(returns_out));
for t = 1:length(returns_out)
    wdcc(:,t)= GMVP(Hdcc(:,:,t));
    wvine(:,t)= GMVP(Hvine(:,:,t));
end

% Second, obtain the portfolio return series for each variance-covariance
% based model
e1 = zeros(length(returns_out),1); e2 = zeros(length(returns_out),1); e3 = zeros(length(returns_out),1);
e4 = zeros(length(returns_out),1); e5 = zeros(length(returns_out),1);
for t = 1:length(returns_out)
    e1(t) = wdcc(:,t)'*returns_out(t,:)';
    e2(t) = wvine(:,t)'*returns_out(t,:)';
    e3(t) = w_ao'*returns_out(t,:)';
    e4(t) = w_sample'*returns_out(t,:)';
    e5(t) = sum(returns_out(t,:))/N;
end

% Out-of-sample performance metrics: average portfolio return and standard
% deviation
average_return = 252*[mean(e1) mean(e2) mean(e3) mean(e4) mean(e5)];
sd_return = sqrt(252)*[std(e1) std(e2) std(e3) std(e4) std(e5)];

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%% DGP 5: scalar DCC correlation process %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
clc
% Specify the desired dimension and full sample size
N = 10; T = 5001;

% Define the univariate GARCH(1,1) processes
hsim2 = zeros(T,N); hsim2(1,:) = 0.005.*ones(1,N);
hsim = zeros(T,N); hsim(1,:) = sqrt(hsim2(1,:));
% Define the variance-covariance and correlation matrices
Sigma = zeros(N,N,T); Correlation = zeros(N,N,T);
% Simulate the univariate GARCH(1,1) parameters (satisfying the
% stationarity constraints)
constant = 0.0001 + (0.009-0.0001)*rand(1,N);
[a_garch,b_garch] = simulate_garch_param(N);
% Define the T x N matrix of observations
returns = zeros(T,N);

% gamma: degree of freedom of the Student distribution
gamma = 3;

cond = true;
while cond
    b = 0.6+(0.9-0.6)*rand(1);
    a = (0.1+(0.2-0.1)*rand(1));
    cond = (any(abs(a+b) > 1));
end

Qbar = simulate_sparse_correlation(N,0);
Qt=zeros(N,N,T);  Qt(:,:,1)=Qbar; stdresid = zeros(T,N);

for t = 2:T
    
    hsim2(t,:) = constant + b_garch.*hsim2(t-1,:) + a_garch.*(returns(t-1,:).^2);
    hsim(t,:) = sqrt(hsim2(t,:));
    
    Qt(:,:,t)=Qbar*(1-a-b) + a*(stdresid(t-1,:)'*stdresid(t-1,:)) + b*Qt(:,:,t-1);
    Correlation(:,:,t)=Qt(:,:,t)./(sqrt(diag(Qt(:,:,t)))*sqrt(diag(Qt(:,:,t)))');
    
    Sigma(:,:,t) = diag(hsim(t,:))*Correlation(:,:,t)*diag(hsim(t,:));
    
    nu_temp = sqrt(gamma-2)*trnd(gamma,1,N)/sqrt(gamma);
    returns(t,:) = (Sigma(:,:,t)^(1/2)*nu_temp')';
    stdresid(t,:) = returns(t,:)./hsim(t,:);
    
end

clear t
% Discard the first matrix of Correlation and the first line of returns
Correlation = Correlation(:,:,2:end); returns = returns(2:end,:); T = length(returns);
%%%%%%%%%%%%%%%%%%%%%%%%%%% In-sample estimation %%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Vine-GARCH model %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%% C-Vine-GARCH is considered
%%%%%%%%%%%%%%  Step 1: select the central node according to AKT
[node,returns] = select_Cvine_akt(returns);

% Hereafter, work with the re-ordered matrix of observations 'returns' to
% estimate the C-Vine GARCH and DCC models

% Alternative: average conditional Kendall's tau
% [node,returns_new] = select_Cvine(returns)

% Alternative: average correlation coefficient
% [node,returns_new] = select_Cvine_corrcoeff(returns);

% Re-ordering of the indices in the true correlation matrix if the interest
% is to compare the true correlation processes with the estimated dcc and
% vine GARCH
for t = 1:T
    Correlation(:,:,t) = Correlation(node,node,t);
end

% Define the in-sample and out-of-sample periods
T_in = 4000; T_out = T_in+1;
returns_in = returns(1:T_in,:); returns_out = returns(T_out:end,:);

%%%%%%%%%%%%%% In-sample estimation
% level: the level up to which the estimation is performed
% level = 2: the partial correlation processes located in the 2 first
% levels of the C-vine tree are estimated; the remaining partial
% correlations are set as their sample partial correlations (computed from
% the sample correlation matrix and the underlying C-vine tree model)
level = 2; method = 'truncation';
[Rt_vine,Ht_vine,parameters_vine,~] = dynamic_vine(returns_in,method,level);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% scalar DCC model %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%% In-sample estimation
[parameters_dcc,Rt_dcc,H_in] = dcc_mvgarch(returns_in,'full');

%%%%%%%%%%%%%%%%%%%%%%%%% Out-of-sample evaluation %%%%%%%%%%%%%%%%%%%%%%%%
% Out-of-sample forecasts for the scalar DCC based
% univariate GARCH(1,1) processes
h_oos = zeros(size(returns_out,1),size(returns_out,2));
index = 1;
for jj=1:size(returns_out,2)
    univariateparameters = parameters_dcc(index:index+1+1);
    [simulatedata,h_oos(:,jj)] = dcc_univariate_simulate(univariateparameters,1,1,returns_out(:,jj));
    index=index+1+1+1;
end
% Generate the out-of-sample conditional GARCH(1,1) variances
h_oos = sqrt(h_oos); % transform into volatility

% scalar DCC out-of-sample correlation process
[~,Rt_dcc_oos,~,~] = dcc_mvgarch_process_oos(parameters_dcc,returns_out,returns_in,H_in);

% C-vine GARCH out-of-sample correlation process
[~,Rt_vine_oos] = Cvine_correlation_process_oos(parameters_vine,returns,T_in,h_oos,returns_out,method,level);

Hdcc = zeros(N,N,length(returns_out));
% scalar DCC out-of-sample variance covariance process
for t = 1:length(returns_out)
    Hdcc(:,:,t) = diag(h_oos(t,:))*Rt_dcc_oos(:,:,t)*diag(h_oos(t,:));
end

% truncated C-vine out-of-sample correlation process
Hvine = zeros(N,N,length(returns_out));
for t = 1:length(returns_out)
    Hvine(:,:,t) = diag(h_oos(t,:))*Rt_vine_oos(:,:,t)*diag(h_oos(t,:));
end

% Average oracle estimator
train = 500; test = 500; B = 10000;
Sigma_ao = average_oracle(returns_in,train,test,B);
w_ao = GMVP(Sigma_ao);

% Sample variance-covariance estimator
w_sample = GMVP(cov(returns_in));

% Computation of the out-of-sample averaged portfolio return and standard
% deviation
% First, obtain the GMVP based portfolio weights
wdcc = zeros(N,length(returns_out)); wvine = zeros(N,length(returns_out));
for t = 1:length(returns_out)
    wdcc(:,t)= GMVP(Hdcc(:,:,t));
    wvine(:,t)= GMVP(Hvine(:,:,t));
end

% Second, obtain the portfolio return series for each variance-covariance
% based model
e1 = zeros(length(returns_out),1); e2 = zeros(length(returns_out),1); e3 = zeros(length(returns_out),1);
e4 = zeros(length(returns_out),1); e5 = zeros(length(returns_out),1);
for t = 1:length(returns_out)
    e1(t) = wdcc(:,t)'*returns_out(t,:)';
    e2(t) = wvine(:,t)'*returns_out(t,:)';
    e3(t) = w_ao'*returns_out(t,:)';
    e4(t) = w_sample'*returns_out(t,:)';
    e5(t) = sum(returns_out(t,:))/N;
end

% Out-of-sample performance metrics: average portfolio return and standard
% deviation
average_return = 252*[mean(e1) mean(e2) mean(e3) mean(e4) mean(e5)];
sd_return = sqrt(252)*[std(e1) std(e2) std(e3) std(e4) std(e5)];


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%% DGP 6: Vine-GARCH correlation model %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear
clc
% Specify the desired dimension and full sample size
N = 10; T = 5001; dim = N*(N-1)/2;

% Define the univariate GARCH(1,1) processes
hsim2 = zeros(T,N); hsim2(1,:) = 0.005.*ones(1,N);
hsim = zeros(T,N); hsim(1,:) = sqrt(hsim2(1,:));
% Define the variance-covariance and correlation matrices
Sigma = zeros(N,N,T); Correlation = zeros(N,N,T); R_partial = zeros(N,N,T);
% Simulate the univariate GARCH(1,1) parameters (satisfying the
% stationarity constraints)
constant = 0.0001 + (0.009-0.0001)*rand(1,N);
[a_garch,b_garch] = simulate_garch_param(N);
% Define the T x N matrix of observations
returns = zeros(T,N);

% gamma: degree of freedom of the Student distribution
gamma = 3;

N1 = count_correl(N,1);
N2 = count_correl(N-1,1); N3 = dim-N1-N2;

alpha_1 = (0.01+(0.2-0.01)*rand(N1,1));
beta_1 = 0.8+(0.99-0.8)*rand(N1,1);
zeta_1 = (0.001+(0.4-0.001)*rand(N1,1));

alpha_2 = (0.01+(0.05-0.01)*rand(N2,1));
beta_2 = 0.8+(0.99-0.8)*rand(N2,1);
zeta_2 = (0.001+(0.1-0.001)*rand(N2,1));


alpha_3 = (0.01+(0.02-0.01)*rand(N3,1));
beta_3 = 0.8+(0.99-0.8)*rand(N3,1);
zeta_3 = (0.001+(0.05-0.001)*rand(N3,1));

rho = zeros(dim,T); rho_tan = zeros(dim,T);
eta = zeros(T,dim);
alpha = [alpha_1;alpha_2;alpha_3]; beta = [beta_1;beta_2;beta_3];
zeta = [zeta_1;zeta_2;zeta_3];


for t = 2:T
    
    hsim2(t,:) = constant + b_garch.*hsim2(t-1,:) + a_garch.*(returns(t-1,:).^2);
    hsim(t,:) = sqrt(hsim2(t,:));
    
    rho_tan(:,t) = alpha + beta.*tan((pi/2).*rho(:,t-1)) + zeta.*eta(t-1,:)';
    rho(:,t) = (2/pi).*atan(rho_tan(:,t));
    R_partial(:,:,t) = vech_off(rho(:,t),N);
    Correlation(:,:,t) = partial2corr_Cvine(R_partial(:,:,t));
    
    
    Sigma(:,:,t) = diag(hsim(t,:))*Correlation(:,:,t)*diag(hsim(t,:));
    
    nu_temp = sqrt(gamma-2)*trnd(gamma,1,N)/sqrt(gamma);
    returns(t,:) = (Sigma(:,:,t)^(1/2)*nu_temp')';
    
    for i = 1:N-1
        eta(t,i)=(returns(t,1)./hsim(t,1)).*(returns(t,i+1)./hsim(t,i+1));
    end
    
    temp = projection(returns(t,:),hsim(t,:),Correlation(:,:,t));
    eta(t,N:end) = temp;
    
end

clear t
% Discard the first matrix of Correlation and the first line of returns
Correlation = Correlation(:,:,2:end); returns = returns(2:end,:); T = length(returns);
%%%%%%%%%%%%%%%%%%%%%%%%%%% In-sample estimation %%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Vine-GARCH model %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%% C-Vine-GARCH is considered
%%%%%%%%%%%%%%  Step 1: select the central node according to AKT
[node,returns] = select_Cvine_akt(returns);

% Hereafter, work with the re-ordered matrix of observations 'returns' to
% estimate the C-Vine GARCH and DCC models

% Alternative: average conditional Kendall's tau
% [node,returns_new] = select_Cvine(returns)

% Alternative: average correlation coefficient
% [node,returns_new] = select_Cvine_corrcoeff(returns);

% Re-ordering of the indices in the true correlation matrix if the interest
% is to compare the true correlation processes with the estimated dcc and
% vine GARCH
for t = 1:T
    Correlation(:,:,t) = Correlation(node,node,t);
end

% Define the in-sample and out-of-sample periods
T_in = 4000; T_out = T_in+1;
returns_in = returns(1:T_in,:); returns_out = returns(T_out:end,:);

%%%%%%%%%%%%%% In-sample estimation
% level: the level up to which the estimation is performed
% level = 2: the partial correlation processes located in the 2 first
% levels of the C-vine tree are estimated; the remaining partial
% correlations are set as their sample partial correlations (computed from
% the sample correlation matrix and the underlying C-vine tree model)
level = 2; method = 'truncation';
[Rt_vine,Ht_vine,parameters_vine,~] = dynamic_vine(returns_in,method,level);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% scalar DCC model %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%% In-sample estimation
[parameters_dcc,Rt_dcc,H_in] = dcc_mvgarch(returns_in,'full');

%%%%%%%%%%%%%%%%%%%%%%%%% Out-of-sample evaluation %%%%%%%%%%%%%%%%%%%%%%%%
% Out-of-sample forecasts for the scalar DCC based
% univariate GARCH(1,1) processes
h_oos = zeros(size(returns_out,1),size(returns_out,2));
index = 1;
for jj=1:size(returns_out,2)
    univariateparameters = parameters_dcc(index:index+1+1);
    [simulatedata,h_oos(:,jj)] = dcc_univariate_simulate(univariateparameters,1,1,returns_out(:,jj));
    index=index+1+1+1;
end
% Generate the out-of-sample conditional GARCH(1,1) variances
h_oos = sqrt(h_oos); % transform into volatility

% scalar DCC out-of-sample correlation process
[~,Rt_dcc_oos,~,~] = dcc_mvgarch_process_oos(parameters_dcc,returns_out,returns_in,H_in);

% C-vine GARCH out-of-sample correlation process
[~,Rt_vine_oos] = Cvine_correlation_process_oos(parameters_vine,returns,T_in,h_oos,returns_out,method,level);

Hdcc = zeros(N,N,length(returns_out));
% scalar DCC out-of-sample variance covariance process
for t = 1:length(returns_out)
    Hdcc(:,:,t) = diag(h_oos(t,:))*Rt_dcc_oos(:,:,t)*diag(h_oos(t,:));
end

% truncated C-vine out-of-sample correlation process
Hvine = zeros(N,N,length(returns_out));
for t = 1:length(returns_out)
    Hvine(:,:,t) = diag(h_oos(t,:))*Rt_vine_oos(:,:,t)*diag(h_oos(t,:));
end

% Average oracle estimator
train = 500; test = 500; B = 10000;
Sigma_ao = average_oracle(returns_in,train,test,B);
w_ao = GMVP(Sigma_ao);

% Sample variance-covariance estimator
w_sample = GMVP(cov(returns_in));

% Computation of the out-of-sample averaged portfolio return and standard
% deviation
% First, obtain the GMVP based portfolio weights
wdcc = zeros(N,length(returns_out)); wvine = zeros(N,length(returns_out));
for t = 1:length(returns_out)
    wdcc(:,t)= GMVP(Hdcc(:,:,t));
    wvine(:,t)= GMVP(Hvine(:,:,t));
end

% Second, obtain the portfolio return series for each variance-covariance
% based model
e1 = zeros(length(returns_out),1); e2 = zeros(length(returns_out),1); e3 = zeros(length(returns_out),1);
e4 = zeros(length(returns_out),1); e5 = zeros(length(returns_out),1);
for t = 1:length(returns_out)
    e1(t) = wdcc(:,t)'*returns_out(t,:)';
    e2(t) = wvine(:,t)'*returns_out(t,:)';
    e3(t) = w_ao'*returns_out(t,:)';
    e4(t) = w_sample'*returns_out(t,:)';
    e5(t) = sum(returns_out(t,:))/N;
end

% Out-of-sample performance metrics: average portfolio return and standard
% deviation
average_return = 252*[mean(e1) mean(e2) mean(e3) mean(e4) mean(e5)];
sd_return = sqrt(252)*[std(e1) std(e2) std(e3) std(e4) std(e5)];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Compute classic correlation from partial correlations based on any arbitrary R-vine
clear
clc
% Example 1
VineArray1 = [ 5 0 0 0 0 ;
    2 2 0 0 0 ;
    3 3 3 0 0 ;
    1 4 4 4 0 ;
    4 1 1 1 1 ];
% Specify a partial correlaton matrix (no need to specify the diagonal
% coefficients)
PCorrelation1 = [ 0 0 0 0 0 ;
    0.2 0 0 0 0 ;
    0.9 0.1 0 0 0 ;
    0.5 0.6 0.7 0 0 ;
    0.8 0.9 0.5 0.8 0 ];

Correlation1 = pcorr2corr(PCorrelation1,VineArray1);

% Verify we obtain PCorrelation1 from Correlation1 based on VineArray1:
corr2pcorr(Correlation1,VineArray1)

% Example 2
VineArray2 = [ 1 0 0 0 ;
    3 3 0 0 ;
    4 4 4 0 ;
    2 2 2 2 ];
PCorrelation2 = [ 0 0 0 0 ;
    0.2 0 0 0 ;
    0 0.2 0 0 ;
    0.6 0.6 0.6 0 ];

Correlation2 = pcorr2corr(PCorrelation2,VineArray2);

% Verify we obtain PCorrelation2 from Correlation2 based on VineArray2:
corr2pcorr(Correlation2,VineArray2)

% Example 3
VineArray3 = [5 0 0 0 0 ;
    2 2 0 0 0 ;
    3 3 3 0 0 ;
    1 4 4 4 0 ;
    4 1 1 1 1 ];
PCorrelation3 = [0 0 0 0 0 ;
    0.2 0 0 0 0 ;
    0.9 0.1 0 0 0 ;
    0.5 0.6 0.9 0 0 ;
    0.9 0.9 0.5 0.8 0];
Correlation3 = pcorr2corr(PCorrelation3,VineArray3);

% Verify we obtain PCorrelation2 from Correlation2 based on VineArray2:
corr2pcorr(Correlation3,VineArray3)