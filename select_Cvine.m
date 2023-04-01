function [node_selection,data] = select_Cvine(data_input)

% C-vine selection procedure based on the average (conditional) Kendall’s
% tau estimator, provided by formula (3.5) in Gijbels, Omelka and 
% Veraverbeke (2015), 'Partial and average copulas and association 
% measures', Electronic Journal of Statistics, Vol. 9, 2420–2474
% The method may not be reliable when the number of variables N in
% data_input is larger than 5 or 6
% See select_Cvine_akt.m and select_Cvine_corrcoeff.m for alternative
% C-vine selection methods (approximation methods)

% Inputs:
%        -  data_input: T x N matrix of observations, where T is the sample
%        size and N is the number of variables

% Outputs:
%        - node_selection: 1 x N vector of the indices of data_input, which
%        are ordered according to the root in each C-vine tree level, i.e.,
%        node_selection(1) is the root of the C-vine in tree level 1,
%        node_selection(2) is the root of the C-vine in tree level 2, etc.
%        - data: data_input re-ordered according to node_selection, i.e.,
%        the first column of data is a T x 1 vector which will be the root 
%        of the C-vine in tree level 1; the second column of data is a 
%        T x 1 vector which will be the root of the C-vine in tree level 2,
%        given the variable in the first column, etc.

data = data_input; [T,N] = size(data); tau = zeros(N,N);
node = zeros(N,1); variable_index = 1:N;
for i = 1:N
    for j = i+1:N
        tau(j,i) = corr(data(:,i),data(:,j),'type','Kendall'); tau(i,j) = tau(j,i);
    end
end
[~,i0] = max(sum(abs(tau))); node(1) = i0;
ynode = data(:,i0); node_selection = variable_index(i0);
data(:,i0) = []; variable_index(i0) = [];

hh = T^(-1/1.8);
for level = 2:N % count over the tree
    fprintf(1,'Selection of the root at level %d\n',level)
    M = size(data,2);
    tau = zeros(M-1,M); y1 = data(:,1);
    for j = 2:M
        y2 = data(:,j);
        kappa = zeros(T,1);
        parfor m = 1:T
            w = normpdf((ynode-ynode(m,:))./hh);
            weight = prod(w,2)/sum(prod(w,2));
            p = zeros(T,1); a1 = y1; a2 = y2;
            for t = 1:T
                p(t) = sum(weight.*((a1(t)-y1<0)&(a2(t)-y2<0)));
            end
            kappa(m) = (4/(1-sum(weight.^2)))*sum(weight.*p) - 1;
        end
        tau(j-1,1) = mean(kappa);
    end
    
    for i = 2:M
        y1 = data(:,i); y = [data(:,1:i-1) data(:,i+1:end)];
        for j = 1:M-1
            y2 = y(:,j);
            kappa = zeros(T,1);
            parfor m = 1:T
                w = normpdf((ynode-ynode(m,:))./hh);
                weight = prod(w,2)/sum(prod(w,2));
                p = zeros(T,1); a1 = y1; a2 = y2;
                for t = 1:T
                    p(t) = sum(weight.*((a1(t)-y1<0)&(a2(t)-y2<0)));
                end
                kappa(m) = (4/(1-sum(weight.^2)))*sum(weight.*p) - 1;
            end
            tau(j,i) =  mean(kappa);
            clear kappa
        end
    end
    [~,i0] = max(sum(tau));
    ynode = [ynode,data(:,i0)]; data(:,i0) = [];
    node_selection = [node_selection,variable_index(i0)]; variable_index(i0) = [];
    node(level) = i0;
    clear i0
end

data = data_input; y = zeros(size(data,1),size(data,2));
for ii = 1:N-2
    y(:,ii) = data(:,node(ii));
    data(:,node(ii)) = [];
end
y(:,ii+1:end) = data; data = y;