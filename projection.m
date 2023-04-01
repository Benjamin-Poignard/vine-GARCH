function eta = projection(data,h,correlation)

% projection.m generates the innovation process of the C-vine GARCH model
% under the parametric assumption (Gaussian variables)

% The variable ordering in data is of key importance to generate the
% innovation process using projection.m
% - the first column in data corresponds to the variable that is the
% central node in the first level of the C-vine tree
% - the second column in data corresponds to the variable that is the
% central node in the second level of the C-vine tree
% - and the like for the third, fourth, ..., (N-1)-th column

% Inputs:   
%        - data: 1 x L vector of observations, with L=level+1, where the 
%        L-2 first columns are the conditioning variables in 'level' of the
%        tree, the L-1 entry is the root variable in tree 'level', and the 
%        L entry forms the edge with the L-1 entry
%        - h: T x L matrix of GARCH(1,1) univariate processes of the
%        entries in the 'data' matrix
%        - correlation: L x L classic correlation matrix of data

% Output:
%        - eta: L-1 x 1 vector of the innovation processes in the L-1
%        entries of the C-vine GARCH process

N = size(data,2); K = zeros(N-1,N-1);
for ii = 2:N-1
    variance = zeros(ii-1,ii-1); R = zeros(ii-1,1);
    for mm = 1:ii-1
        for pp = 1:ii-1
            variance(mm,pp) = correlation(mm,pp)*h(mm)*h(pp);
        end
        clear pp
        R(mm,1) = data(mm);
    end
    clear mm
    f = [];
    for jj = ii:N
        c = [];
        for kk = 1:ii-1
            c = [ c correlation(kk,jj)*h(kk)*h(jj) ];
        end
        clear kk
        f = [f,(data(jj)-c*inv(variance)*R)/sqrt(h(jj)^2-c*inv(variance)*c')];
    end
    clear jj
    K(ii-1,ii-1:end) = f;
    clear f R
end
clear ii

e = zeros(N-1,N-1);
for ll = 1:N-1
    for nn = ll+1:N-1
        e(ll,nn) = K(ll,ll)*K(ll,nn);
    end
    clear nn
end
clear ll
eta = vech_on(tril(e'),N-1)';