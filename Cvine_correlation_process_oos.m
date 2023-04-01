function [PCorrelation,Correlation] = Cvine_correlation_process_oos(param,data,T_in,h_oos,data_oos,method,level)

% Cvine_correlation_process_oos.m generates the out-of-sample C-vine GARCH
% partial correlation and classi correlation matrix processes, with/without
% truncation

% Inputs:
%        - param: N*(N-1)/2 x 3 matrix of the C-vine GARCH partial
%        correlation processes, with N the dimension of the vector of
%        observations. If the model is truncated, then the coefficients
%        in param from the truncation level are zero. 
%        Each of the N-1 first lines of params contains the three 
%        parameters of the corresponding partial correlation process 
%        (classic correlations in the first tree level) located in the 
%        first tree level: line 1 corresponds to the edge(1,2), line 2 for 
%        the edge (1,3), ..., line N-1 for (1,N). From line N to 2*N-3, the
%        lines represent the coefficients for the edges (2,3|1), (2,4|1), 
%        ..., (2,N|1). And the like for the next tree levels
%        - data: full data set 
%        - T_in: last date of the in-sample period
%        - h_oos: T_out x N out-of-sample matrix of the univariate
%        volatility processes 
%        - data_oos: T_out x N out-of-sample matrix of the vector of
%        observations
%        - method: 'truncation' or 'no-truncation'
%        ==> if 'truncation': truncated C-vine model, where the 
%        estimation was performed up to the specified level; the remaining
%        partial correlations are set as their sample partial correlations
%        when generating the C-vine GARCH correlation process (computed
%        from the sample correlation matrix and the underlying C-vine tree 
%        model)
%        - level: optional if method = 'no-truncation'; required if
%        method = 'truncation'; the level of the C-vine tree up to which
%        the estimation is performed

% Outputs:
%        - PCorrelation: N x N x T_out partial correlation process
%        generated from the (truncated) C-vine GARCH process, with T_out
%        the size of the out-of-sample period
%        - Correlation: N x N x T_out correlation process generated from 
%        the (truncated) C-vine GARCH process, obtained from 
%        partial2corr_Cvine.m

% If param is provided as a vector, then convert it to a N*(N-1)/2 x 3
% matrix
[T,N] = size(data_oos);
if size(param,2)<2
    param = reshape(param,N*(N-1)/2,3);
end

M = zeros(N*(N-1)/2,T-1); M(:,1) = vech_on(corr2partial_Cvine(corrcoef(data(1:T_in,:))),N);
M_tan = zeros(N*(N-1)/2,T-1); PCorrelation = zeros(N,N,T-1);
Correlation = zeros(N,N,T); Correlation(:,:,1) = corrcoef(data(1:T_in,:));

if N==2
    eta =((data_oos(:,1)-mean(data_oos(:,1)))./h_oos(:,1)).*((data_oos(:,2)-mean(data_oos(:,2)))./h_oos(:,2));
    for t = 2:T
        M_tan(:,t) = param(:,1) + param(:,2).*tan((pi/2).*M(:,t-1)) + param(:,3).*eta(t-1,:)';
        M(:,t) = (2/pi).*atan(M_tan(:,t));
        PCorrelation(:,:,t) = vech_off(M(:,t),N);
        Correlation(:,:,t) = partial2corr_Cvine(PCorrelation(:,:,t));
    end
else
    eta = zeros(T,N*(N-1)/2);
    for i = 1:N-1
        eta(:,i)=((data_oos(:,1)-mean(data_oos(:,1)))./h_oos(:,1)).*((data_oos(:,i+1)-mean(data_oos(:,i+1)))./h_oos(:,i+1));
    end
    clear i
    switch method
        case 'no-truncation'
            eta(1,N:end) = projection(data_oos(1,:),h_oos(1,:),Correlation(:,:,1));
            for t = 2:T
                M_tan(:,t) = param(:,1) + param(:,2).*tan((pi/2).*M(:,t-1)) + param(:,3).*eta(t-1,:)';
                M(:,t) = (2/pi).*atan(M_tan(:,t));
                PCorrelation(:,:,t) = vech_off(M(:,t),N);
                Correlation(:,:,t) = partial2corr_Cvine(PCorrelation(:,:,t));
                temp = projection(data_oos(t,:),h_oos(t,:),Correlation(:,:,t));
                eta(t,N:end) = temp;
                clear temp
            end
        case 'truncation'
            M_temp = corr2partial_Cvine(corrcoef(data(1:T_in,:)));
            eta_temp = [];
            for ii=2:level
                for jj=ii+1:N
                    correl_temp = eye(ii+1); correl_temp(1:ii,1:ii) = Correlation(1:ii,1:ii,1);
                    correl_temp(1:ii,ii+1) = Correlation(1:ii,jj,1); correl_temp(ii+1,1:ii) = correl_temp(1:ii,end);
                    eta_t = projection([data_oos(1,1:ii),data_oos(1,jj)],[h_oos(1,1:ii),h_oos(1,jj)],correl_temp);
                    eta_temp = [eta_temp;eta_t(end)];
                end
            end
            eta(1,N:count_correl(N,level)) = eta_temp;
            for t = 2:T
                M_tan(:,t) = param(:,1) + param(:,2).*tan((pi/2).*M(:,t-1)) + param(:,3).*eta(t-1,:)';
                M(:,t) = (2/pi).*atan(M_tan(:,t));
                PCorrelation(:,:,t) = vech_off(M(:,t),N);
                PCorrelation(level+1:end,level+1:end,t) = M_temp(level+1:end,level+1:end);
                Correlation(:,:,t) = partial2corr_Cvine(PCorrelation(:,:,t));
                eta_temp = [];
                for ii=2:level
                    for jj=ii+1:N
                        correl_temp = eye(ii+1); correl_temp(1:ii,1:ii) = Correlation(1:ii,1:ii,t);
                        correl_temp(1:ii,ii+1) = Correlation(1:ii,jj,t); correl_temp(ii+1,1:ii) = correl_temp(1:ii,end);
                        eta_t = projection([data_oos(t,1:ii),data_oos(t,jj)],[h_oos(t,1:ii),h_oos(t,jj)],correl_temp);
                        eta_temp = [eta_temp;eta_t(end)];
                    end
                end
                eta(t,N:count_correl(N,level)) = eta_temp;
                clear eta_temp
            end
            clear t
    end
end