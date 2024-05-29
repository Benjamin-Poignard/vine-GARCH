function C = simulate_sparse_correlation(p,threshold)

% p: dimension
% sparsity: sparsity degree, precentage of p*(p-1)/2

dim = p*(p-1)/2;
cond = true;
while cond
    temp = round(rand(dim,1)); temp(temp==0)=-1;
    C = vech_off((rand(dim,1)>threshold).*temp.*(0.005+0.5*rand(dim,1)),p);
    beta_true = vech_on(C,p);
    count = 0;
    for ii = 1:dim
        if (beta_true(ii)==0)
            count = count+1;
        else
            count = count+0;
        end
    end
    cond = (min(eig(C))<0.01);
end
