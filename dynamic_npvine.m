function [Correlation,VarCov,parameters_vine,parameters_raw] = dynamic_npvine(data,method,level,window_size)

% C-vine GARCH estimation, with underlying structure given by the C-vine
% tree structure, and with non-parametric innovation process

% The C-vine GARCH generates dynamic partial correlations; classic
% correlations are obtained using the mapping partial2corr_Cvine.m, which
% generates classic correlations from partial correlations based on the
% C-vine tree

% The variable ordering in data is of key importance:
% - the first column in data corresponds to the variable that is the
% central node in the first level of the C-vine tree
% - the second column in data corresponds to the variable that is the
% central node in the second level of the C-vine tree
% - and the like for the third, fourth, ..., (N-1)-th column

% The estimation is performed level-by-level in the C-vine tree, and
% parallelization w.r.t. each couple located in the corresponding level is
% employed: a 'local' log-likelihood is optimized for each couple

% Inputs:
%        - data: T x N vector of observations
%        - method: 'truncation' or 'no-truncation'
%        ==> if 'truncation': truncated C-vine estimation, so that the
%        estimation is performed up to the specified level; the remaining
%        partial correlations are set as their sample partial correlations
%        (computed from the sample correlation matrix and the underlying
%        C-vine tree model)
%        ==> if 'no-truncation': the partial correlations for all C-vine
%        tree levels are estimated, no-truncation
%        - level: optional if method = 'no-truncation'; required if
%        method = 'truncation'; the level of the C-vine tree up to which
%        the estimation is performed; Note: level>1
%        - window_size: length of the rolling window to generate the
%        innovations

% Outputs:
%        - Correlation: N x N x T dynamic correlation matrix process
%        generated from the (truncated) C-vine structure under the
%        parametric assumption (conditionally Gaussian variables)
%        - VarCov: N x N x T dynamic covariance matrix process
%        generated from the (truncated) C-vine structure under the
%        parametric assumption (conditionally Gaussian variables) and with
%        GARCH(1,1) univariate conditional variance processes
%        - parameters_vine: N*(N-1)/2 x 3 matrix
%        - parameters_raw: 3*N+3*N*(N-1)/2 x 1 vector containing all the
%        estimated parameters of the C-vine GARCH model (i.e., stacking the
%        estimated parameters of the N GARCH(1,1) univariate processes and
%        parameters_vine)

[T,N]=size(data);
if level<2
    error('At least two tree levels must be estimated')
end
% Step 1: estimation of the univariate GARCH(1,1) processes
opts.Hessian = 'bfgs';
opts.MaxRLPIter = 100000;
opts.Algorithm = 'sqp';
opts.TolCon = 1e-09;
opts.TolRLPFun = 1e-09;
opts.MaxSQPIter = 100000;
h = zeros(T,N); u = zeros(T,N); parameters_univ = zeros(3,N); MatAsym = zeros(3,3,N); Score = zeros(T,3,N);
for i=1:N
    [param,~,~,robustSE,ht,scores] = fattailed_garch(data(:,i),1,1,'NORMAL',[],opts);
    u(:,i)=data(:,i)./sqrt(ht); parameters_univ(:,i) = param; MatAsym(:,:,i) = robustSE;
    h(:,i) = sqrt(ht); Score(:,:,i) = scores;
end

% Step 2: estimation of the parameters of the dynamic correlation process
opts.MaxRLPIter = 500000;
opts.MaxFunEvals = 500000;
opts.Algorithm = 'sqp';
opts.TolCon = 1e-09;
opts.TolRLPFun = 1e-09;
opts.MaxSQPIter = 500000;
opts.Jacobian = 'off';
opts.Display = 'off';

% First tree level
if N==2
    
    cpartial = []; coeff = [];
    parfor ii = 2:N
        [param,~,~,~,~,~]=fmincon(@(x)reduced_likelihood(x,[data(:,1),data(:,ii)],[h(:,1),h(:,ii)]),[0.001+(0.05-0.001)*rand(1),0.85+(0.95-0.85)*rand(1),0.01+(0.02)*rand(1)],[],[],[],[],[],[],@(x)vine_constr_tree1(x),opts);
        [~,~,correl] = reduced_likelihood(param,[data(:,1),data(:,ii)],[h(:,1),h(:,ii)]);
        cpartial = [cpartial,squeeze(correl(1,2,:))];
        coeff = [coeff;param];
    end
    clear ii
    fprintf(1,'Estimation of the partial correlation processes in the first tree completed \n')
    % Fill the full partial correlation matrix from the univariate partial
    % correlation processes
    Correlation_partial = zeros(N,N,T);
    for t = 1:T
        Correlation_partial(:,:,t) = eye(N,N);
        Correlation_partial(1,2:end,t) = cpartial(t,1:N-1);
        Correlation_partial(2:end,1,t) = Correlation_partial(1,2:end,t);
    end
    clear t
    
else
    
    cpartial = []; coeff = [];
    parfor ii = 2:N
        [param,~,~,~,~,~]=fmincon(@(x)reduced_likelihood(x,[data(:,1),data(:,ii)],[h(:,1),h(:,ii)]),[0.001+(0.05-0.001)*rand(1),0.85+(0.95-0.85)*rand(1),0.01+(0.02-0.01)*rand(1)],[],[],[],[],[],[],@(x)vine_constr_tree1(x),opts);
        [~,~,correl] = reduced_likelihood(param,[data(:,1),data(:,ii)],[h(:,1),h(:,ii)]);
        cpartial = [cpartial,squeeze(correl(1,2,:))];
        coeff = [coeff;param];
    end
    clear ii
    fprintf(1,'Estimation of the partial correlation processes in the first tree completed \n')
    Correlation_partial = zeros(N,N,T);
    for t = 1:T
        Correlation_partial(:,:,t) = eye(N,N);
        Correlation_partial(1,2:end,t) = cpartial(t,1:N-1);
        Correlation_partial(2:end,1,t) = Correlation_partial(1,2:end,t);
    end
    clear t
    % Fill the full partial correlation matrix from the univariate partial
    % correlation processes
    ii = 2; ll = [];
    parfor jj = ii+1:N
        kappa = [];
        correl_temp = zeros(ii+1,ii+1,T);
        for t = 1:T
            correl_temp(:,:,t) = eye(ii+1,ii+1);
            correl_temp(1:ii-1,ii:end,t) = [Correlation_partial(1,ii,t),Correlation_partial(1,jj,t)];
            correl_temp(2:end,1,t) = correl_temp(1,2:end,t);
            kappa = [kappa,vech_on(correl_temp(:,:,t),size(correl_temp(:,:,t),1))];
        end
        [v1,v2] = generate_innovation(data,[ii,jj],1:ii-1,window_size);
        [param,~,~,~,~,~]=fmincon(@(x)reduced_likelihood_pcorr_np(x,ii,[data(:,1:ii),data(:,jj)],[h(:,1:ii),h(:,jj)],v1.*v2,kappa,window_size),[0.001+(0.05-0.001)*rand(1),0.85+(0.95-0.85)*rand(1),0.01+(0.02-0.01)*rand(1)],[],[],[],[],[],[],@(x)vine_constr(x),opts);
        [~,~,~,partial] = reduced_likelihood_pcorr_np(param,ii,[data(:,1:ii),data(:,jj)],[h(:,1:ii),h(:,jj)],v1.*v2,kappa,window_size);
        coeff = [coeff;param]; ll = [ll,partial];
    end
    clear start
    fprintf(1,'Estimation of the partial correlation processes in tree %d completed \n',2)
    for t = 1:T
        Correlation_partial(ii,ii+1:end,t) = ll(t,:);
        Correlation_partial(ii+1:end,ii,t) = Correlation_partial(ii,ii+1:end,t);
    end
    clear t ll
    
    switch method
        
        case 'no-truncation'
            for ii = 3:N-1
                ll = [];
                parfor jj = ii+1:N
                    kappa = [];
                    correl_temp = zeros(ii+1,ii+1,T); kappa = [];
                    for t = 1:T
                        correl_temp(:,:,t) = eye(ii+1,ii+1);
                        temp = [Correlation_partial(1:ii-1,ii,t),Correlation_partial(1:ii-1,jj,t)];
                        correl_temp(1:ii-2,2:ii-1,t) = Correlation_partial(1:ii-2,2:ii-1,t);
                        correl_temp(1:ii-1,ii:end,t) = temp;
                        kappa = [kappa,vech_on(correl_temp(:,:,t)',size(correl_temp(:,:,t),1))];
                    end
                    [v1,v2] = generate_innovation(data,[ii,jj],1:ii-1,window_size);
                    [param,~,~,~,~,~]=fmincon(@(x)reduced_likelihood_pcorr_np(x,ii,[data(:,1:ii),data(:,jj)],[h(:,1:ii),h(:,jj)],v1.*v2,kappa,window_size),[0.001+(0.05-0.001)*rand(1),0.85+(0.95-0.85)*rand(1),0.01+(0.02-0.01)*rand(1)],[],[],[],[],[],[],@(x)vine_constr(x),opts);
                    [~,~,~,ppartial] = reduced_likelihood_pcorr_np(param,ii,[data(:,1:ii),data(:,jj)],[h(:,1:ii),h(:,jj)],v1.*v2,kappa,window_size);
                    coeff = [coeff;param]; ll =[ll,ppartial];
                end
                fprintf(1,'Estimation of the partial correlation processes in tree %d completed \n',ii)
                for t = 1:T
                    Correlation_partial(ii,ii+1:end,t) = ll(t);
                    Correlation_partial(ii+1:end,ii,t) = Correlation_partial(ii,ii+1:end,t);
                end
                clear t ll
            end
            clear ii
            
        case 'truncation'
            if level==2
                temp = corr2partial_Cvine(corrcoef(data));
                ii = level;
                for t = 1:T
                    Correlation_partial(ii+1:end,ii+1:end,t) = temp(ii+1:end,ii+1:end);
                end
                clear temp
            else
                for ii = 3:level
                    ll = [];
                    parfor jj = ii+1:N
                        kappa = [];
                        correl_temp = zeros(ii+1,ii+1,T); kappa = [];
                        for t = 1:T
                            correl_temp(:,:,t) = eye(ii+1,ii+1);
                            temp = [Correlation_partial(1:ii-1,ii,t),Correlation_partial(1:ii-1,jj,t)];
                            correl_temp(1:ii-2,2:ii-1,t) = Correlation_partial(1:ii-2,2:ii-1,t);
                            correl_temp(1:ii-1,ii:end,t) = temp;
                            kappa = [kappa,vech_on(correl_temp(:,:,t)',size(correl_temp(:,:,t),1))];
                        end
                        [v1,v2] = generate_innovation(data,[ii,jj],1:ii-1,window_size);
                        [param,~,~,~,~,~]=fmincon(@(x)reduced_likelihood_pcorr_np(x,ii,[data(:,1:ii),data(:,jj)],[h(:,1:ii),h(:,jj)],v1.*v2,kappa,window_size),[0.001+(0.05-0.001)*rand(1),0.85+(0.95-0.85)*rand(1),0.01+(0.02-0.01)*rand(1)],[],[],[],[],[],[],@(x)vine_constr(x),opts);
                        [~,~,~,partial] = reduced_likelihood_pcorr_np(param,ii,[data(:,1:ii),data(:,jj)],[h(:,1:ii),h(:,jj)],v1.*v2,kappa,window_size);
                        coeff = [coeff;param];
                        ll = [ll,partial];
                    end
                    fprintf(1,'Estimation of the partial correlation processes in tree %d completed \n',ii)
                    for t = 1:T
                        Correlation_partial(ii,ii+1:end,t) = ll(t,:);
                        Correlation_partial(ii+1:end,ii,t) = Correlation_partial(ii,ii+1:end,t);
                    end
                    clear t ll
                end
                clear ii
                % The partial correlation components located above the
                % truncation level are set as their sample partial
                % correlations, which are deduced from the correlation
                % matrix using corr2partial_Cvine.m
                temp = corr2partial_Cvine(corrcoef(data));
                ii = level;
                for t = 1:T
                    Correlation_partial(ii+1:end,ii+1:end,t) = temp(ii+1:end,ii+1:end);
                end
                clear temp
            end
    end
    
end
coeff = vec(coeff)'; parameters_raw = [vec(parameters_univ);coeff'];
Correlation = zeros(N,N,T);
for t = 1:T
    Correlation(:,:,t) = partial2corr_Cvine(Correlation_partial(:,:,t));
end
switch method
    case 'truncation'
        parameters_correl_parametric = parameters_raw(3*N+1:end);
        param_non_constr = reshape(parameters_correl_parametric,count_correl(N,level),3);
        temp = vech_on(corr2partial_Cvine(corrcoef(data)),N);
        param_constr = temp(count_correl(N,level)+1:end);
        parameters_vine = [param_non_constr;zeros(length(param_constr),3)];
    case 'no-truncation'
        parameters_vine = reshape(parameters_raw(3*N+1:end),N*(N-1)/2,3);
end

parameters_raw = [vec(parameters_univ);vec(parameters_vine)];

% Generate the dynamic variance-covariance matrix based on the C-vine GARCH
% process
% Step 1: generate the univariate GARCH(1,1) processes
h = zeros(T,N); index = 1;
for i=1:size(data,2)
    univariateparameters = parameters_raw(index:index+1+1);
    [~,h(:,i)] = dcc_univariate_simulate(univariateparameters,1,1,data(:,i));
    index=index+1+1+1;
end
% Step 2: recompose the variance-covariance matrix
VarCov = zeros(N,N,T);
for t = 1:T
    VarCov(:,:,t) = diag(h(t,:))*Correlation(:,:,t)*diag(h(t,:));
end